@echo off
chcp 65001 >nul
REM ============================================================
REM  10B LLM codebase - end-to-end demo (Windows, CPU OK)
REM  Runs the whole pipeline on sample data:
REM  tokenizer -> data -> pretrain -> SFT -> chat
REM  (tiny toy model; answers are low quality, pipeline check only)
REM ============================================================
cd /d %~dp0

echo.
echo === [1/6] Train tokenizer (chat special tokens included) ===
python scripts\train_tokenizer.py --input data\sample_corpus.jsonl --vocab-size 2000 --out tokenizer\
if errorlevel 1 goto err

echo.
echo === [2/6] Tokenize corpus -^> data\train.bin ===
python scripts\prepare_data.py --input data\sample_corpus.jsonl --tokenizer tokenizer\ --out data\train.bin
if errorlevel 1 goto err

echo.
echo === [3/6] Small pretraining (200 steps, CPU) ===
python scripts\train.py --config configs\demo.json --data data\train.bin --out runs\demo
if errorlevel 1 goto err

echo.
echo === [4/6] SFT on chat data (600 steps, overfit tiny data) ===
python scripts\finetune.py --config configs\demo.json --sft-data data\sample_chat.jsonl --tokenizer tokenizer\ --init-from runs\demo\ckpt_final.pt --out runs\demo-sft --max-steps 600
if errorlevel 1 goto err

echo.
echo === [5/6] Build training-curve dashboard ===
python scripts\dashboard.py --metrics runs\demo-sft\metrics.jsonl --out runs\demo-sft\dashboard.html

echo.
echo === [6/6] Start chatting!  (type /exit to quit) ===
echo Tip: try a trained question like:  3 곱하기 7은?
python scripts\chat.py --ckpt runs\demo-sft\ckpt_final.pt --tokenizer tokenizer\ --temperature 0.2 --top-k 10
goto end

:err
echo.
echo [ERROR] A previous step failed. See the message above.
exit /b 1

:end
