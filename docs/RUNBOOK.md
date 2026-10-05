# pixel-clouds 運用・復旧手順書

最終更新: 2026-10-05。Claude や GitHub が使えなくなっても、この1枚で回せるように書いてある。
置き場所: `~/wallpaper-work/RUNBOOK.md` と、リポジトリの `docs/RUNBOOK.md`（同じ内容）。

---

## 0. 全体像

```
NOAA GMGSI(赤外の全球雲)  ─┐
ひまわり9号(AWS)          ─┤  GitHub Actions「Update clouds」(3時間ごと, UTC 毎時17分)
NASA GIBS(北極の差し替え) ─┘
   make_clouds.py --polar        GMGSI→世界地図(clouds_src.png)。緯度はメルカトル補正済み
   himawari_seam.py              93.1°E の衛星境界をひまわりで混ぜる(失敗時は自動スキップ)
   cloud_match.py                純正の雲の明るさ分布に寄せる(暗くする方向だけ)
   clouds_pipeline.py            キューブ6面×16枚=96枚の .dds タイル
   check_seams.py                継ぎ目の検査。NGならその回は配信しない(前回の雲が残る)
   check_faces_diag.py           斜めの線の検査(警告だけ)
   → コミット → GitHub Pages で公開
配信: https://hdhsksdhsk.github.io/pixel-clouds/root.json  → 最新タイルの場所を指す
APK: 2017版(com.breel.wallpapers) と BReel版(com.breel.geswallpapers) がこの URL を読む
```

- 作業フォルダ: `~/wallpaper-work/clouds_output/`（= リポジトリ hdhsksdhsk/pixel-clouds のクローン。全履歴が入っている）
- ワークフロー: `.github/workflows/update.yml`

---

## 1. 状態の確認（ときどき）

```bash
curl -s https://hdhsksdhsk.github.io/pixel-clouds/root.json
```
`baseUrl` の `YYYYMMDD_HHMM`（UTC）が配信中の雲。数日前のままなら止まっている。

```bash
cd ~/wallpaper-work/clouds_output && gh workflow list -R hdhsksdhsk/pixel-clouds --all && gh run list -R hdhsksdhsk/pixel-clouds --workflow "Update clouds" --limit 5 --json createdAt,conclusion,event --jq '.[] | "\(.createdAt)  \(.conclusion)  \(.event)"'
```
ブラウザなら Actions → 最新の実行 → **Annotations に警告が無いか**も見る。

---

## 2. 止まったときの見分け方

| 症状 | 原因 | 対処 |
|---|---|---|
| Update clouds が `disabled_...` | 60日動きが無い等で自動停止 | `gh workflow enable "Update clouds" -R hdhsksdhsk/pixel-clouds && gh workflow run "Update clouds" -R hdhsksdhsk/pixel-clouds` |
| `failure` が続く | 下のログで判断 | ↓ |
| success だが間が空く | GitHub の cron の遅れ・飛ばし | `gh workflow run "Update clouds" -R hdhsksdhsk/pixel-clouds` |
| 自分の push ができない | gh のログイン切れ | `gh auth status` → `gh auth login` |

失敗のログ:
```bash
cd ~/wallpaper-work/clouds_output && gh run view $(gh run list -R hdhsksdhsk/pixel-clouds --workflow "Update clouds" --limit 1 --json databaseId --jq '.[0].databaseId') -R hdhsksdhsk/pixel-clouds --log-failed | tail -40
```
- `check_seams` の NG → 境界が強い日にガードが止めているだけ。数回で戻る。何日も続くときだけ調べる
- 404 などダウンロード失敗 → NOAA/AWS 側のファイル名・場所の変更。スクリプト修正が必要
- `pip install` 失敗 → Python や部品のバージョン問題

---

## 3. 手入れ（警告が出たとき）

| 対象 | 今 | きっかけ |
|---|---|---|
| OS | `runs-on: ubuntu-24.04` | 廃止予定の警告 |
| アクション | checkout@v5 / setup-python@v6 / upload-artifact@v6 | Node.js 非推奨の警告 |
| Python | 3.11 | 2027年10月のサポート終了前 |
| 部品 | numpy pillow quicktex netCDF4 scipy（版指定なし） | 突然の失敗 |

手順（例: OS を上げる）:
```bash
cd ~/wallpaper-work/clouds_output && sed -i '' 's/ubuntu-24.04/ubuntu-26.04/' .github/workflows/update.yml && grep -n "runs-on\|uses:\|python-version" .github/workflows/update.yml
cd ~/wallpaper-work/clouds_output && git add .github/workflows/update.yml && git commit -m "update workflow" && git pull --rebase && git push
cd ~/wallpaper-work/clouds_output && gh workflow run "Update clouds" -R hdhsksdhsk/pixel-clouds
```
確認: success / 警告が消えた / Set up job の Runner Image。だめなら:
```bash
cd ~/wallpaper-work/clouds_output && git revert --no-edit HEAD && git push
```

---

## 4. 処理を外す（戻す）

| 外したいもの | コマンド（clouds_output で実行 → コミット → push） |
|---|---|
| ひまわり＋寄せる処理 | `python3 patch_workflow.py --revert` |
| 緯度の補正 | `python3 patch_merc.py --revert` |
| 北極の差し替え | `python3 ../polar_test/patch_polar.py revert` |

---

## 5. git のつまずき

- push が rejected → Actions が先行しているだけ。`git pull --rebase && git push`
- `??` のファイル → 管理外。pull の邪魔にならない
- rebase の衝突 → 消す側を採るなら `git rm` → `GIT_EDITOR=true git rebase --continue`。迷ったら `git rebase --abort`
- `M root.json` → 手元で焼いた名残。`git checkout -- root.json`
- 手順は `&&` でつなぐ（前が失敗したら止まる）

---

## 6. 端末に反映させる

アプリは雲を覚えているので、すぐ見たいときはキャッシュを消して壁紙を設定し直す（壁紙設定も消える）:
```bash
adb shell pm clear com.breel.geswallpapers
adb shell pm clear com.breel.wallpapers
```

---

## 7. GitHub アカウントを失ったときの復旧

予防: **2段階認証のリカバリーコードをオフラインに保存**。登録メールにログインできる状態を保つ。

復旧の流れ:
1. 新しい GitHub アカウントを作り、空のリポジトリ `pixel-clouds` を作る
2. 手元のクローンを送る
   ```bash
   cd ~/wallpaper-work/clouds_output && git remote set-url origin https://github.com/新ユーザー名/pixel-clouds.git && git push -u origin main
   ```
3. `update.yml` の中の配信URL `https://hdhsksdhsk.github.io/pixel-clouds/` を新しい名前に置き換えてコミット・push
4. リポジトリの Settings → Pages を、**下の「現在のPages設定」と同じ**にする。Settings → Actions で実行を許可
5. `gh workflow run "Update clouds"` → `https://新ユーザー名.github.io/pixel-clouds/root.json` が開けることを確認
6. APK 側の URL を書き換えてビルドし直す
   ```bash
   cd ~/wallpaper-work && grep -rl "hdhsksdhsk.github.io" decoded*/smali* 2>/dev/null
   ```
   出てきたファイルの `hdhsksdhsk` を新しい名前に置き換え → 各 APK をビルド・署名・インストール

現在のPages設定（記録用。下のコマンドで取り直せる）:
```bash
gh api repos/hdhsksdhsk/pixel-clouds/pages --jq '{source: .source, url: .html_url}'
```
→ ここに結果を貼っておく:

---

## 8. Claude が使えなくなったとき

- この手順書と、`~/wallpaper-work/` 内のスクリプト・バックアップがあれば回せる
- 雲以外の壁紙改造の経緯は、作業フォルダ内の各 `*.py` の先頭の説明と `keep/` の完成版 APK が手がかり

## 9. バックアップ（定期的に外付けへ）

最低限: `~/wallpaper-work/keep/`（完成版APK）、`decoded_*`（作業ツリー）、`clouds_output/`（リポジトリ）、4K素材（`*_strip_4k*.png`）、`breel_tex/`、`debug.keystore`（**これを失うと今のAPKを上書き更新できなくなる**）
