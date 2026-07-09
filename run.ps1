param([string]$Command = "help")

$SERVICE     = "app"
$SRC         = "/workspace/src"
$DATA_DIR    = "/workspace/scinti_segmentation"
$PRETRAINED  = "/workspace/uniconvnet_t_1k_224_ema.pth"
$BATCH_SIZE  = 8
$NUM_WORKERS = 4
$MAX_EPOCHS  = 50
$LR          = "1e-4"

function Invoke-Up { docker compose up -d }

switch ($Command) {
    "build" {
        docker compose build
    }
    "up" {
        Invoke-Up
    }
    "down" {
        docker compose down
    }
    "gpu" {
        Invoke-Up
        docker compose exec $SERVICE python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
    }
    "train" {
        Invoke-Up
        docker compose exec $SERVICE python3 "$SRC/train.py" `
            --data-dir    $DATA_DIR `
            --pretrained  $PRETRAINED `
            --batch-size  $BATCH_SIZE `
            --num-workers $NUM_WORKERS `
            --max-epochs  $MAX_EPOCHS `
            --lr          $LR
    }
    "train-freeze" {
        Invoke-Up
        docker compose exec $SERVICE python3 "$SRC/train.py" `
            --data-dir      $DATA_DIR `
            --pretrained    $PRETRAINED `
            --batch-size    $BATCH_SIZE `
            --num-workers   $NUM_WORKERS `
            --max-epochs    $MAX_EPOCHS `
            --lr            $LR `
            --freeze-backbone
    }
    "shell" {
        Invoke-Up
        docker compose exec $SERVICE bash
    }
    default {
        Write-Host "Usage: .\run.ps1 <command>"
        Write-Host "  build         Build Docker image"
        Write-Host "  up            Start container in background"
        Write-Host "  down          Stop and remove container"
        Write-Host "  gpu           Check GPU availability"
        Write-Host "  train         Run training (all params)"
        Write-Host "  train-freeze  Run training (backbone frozen)"
        Write-Host "  shell         Open bash inside container"
    }
}
