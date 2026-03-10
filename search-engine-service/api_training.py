"""
Training API for Airflow Integration.
Simple API to trigger model training via HTTP.

Run: uvicorn api_training:app --port 8001
"""

import asyncio
import json
import logging
import os
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def tee_output(
    process: subprocess.Popen, log_file, job_id: str, stage: str = "training"
):
    """Read process output line-by-line and write to both file and console.

    Runs in daemon thread - dies when parent process dies.
    Output is logged with job_id prefix for easy identification.

    Args:
        process: Subprocess.Popen instance with stdout=PIPE and text=True
        log_file: Already opened file handle (caller must close)
        job_id: Job identifier for log prefix
        stage: Optional stage identifier (e.g., "stage1", "stage2")
    """
    prefix = f"[{job_id}]" if not stage else f"[{job_id}:{stage}]"

    try:
        for line in iter(process.stdout.readline, ""):  # text=True returns strings
            if not line:
                break
            # Write to file
            log_file.write(line)
            log_file.flush()
            # Write to uvicorn console
            logger.info(f"{prefix} {line.strip()}")
    except Exception as e:
        logger.error(f"{prefix} Tee output error: {e}")


# Directories
JOBS_DIR = Path("runtime/jobs")
JOBS_DIR.mkdir(exist_ok=True, parents=True)
CHECKPOINTS_DIR = Path("runtime/checkpoints")

# Track all active training PIDs for graceful shutdown
ACTIVE_TRAINING_PIDS: dict[str, int] = {}  # {job_id: pid}


# ============================================================================
# SCHEMAS
# ============================================================================


class TrainRequest(BaseModel):
    """Training request - just dates."""

    start_date: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2025-01-01"]
    )
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2025-06-30"])


class JobResponse(BaseModel):
    """Job created response."""

    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Job status response."""

    job_id: str
    status: str  # queued, running, completed, failed
    stage: int | str  # 1, 2, or "full" for unified pipeline
    substage: Optional[str] = None  # "stage1" or "stage2" when stage="full"
    start_date: str
    end_date: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_minutes: Optional[float] = None
    output_path: Optional[str] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None
    promotion: Optional[dict] = None


# ============================================================================
# JOB MANAGEMENT (Simple file-based)
# ============================================================================


def create_job(stage: int | str, start_date: str, end_date: str) -> str:
    """Create a new job and return job_id.

    Args:
        stage: 1, 2, or "full" for unified pipeline
    """
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    job_data = {
        "job_id": job_id,
        "status": "queued",
        "stage": stage,
        "start_date": start_date,
        "end_date": end_date,
        "created_at": datetime.now().isoformat(),
    }
    job_file = JOBS_DIR / f"{job_id}.json"
    job_file.write_text(json.dumps(job_data, indent=2))

    # Auto-cleanup: Keep only last 5 jobs to prevent clutter
    cleanup_old_jobs(limit=5)

    return job_id


def cleanup_old_jobs(limit: int = 5):
    """Delete old job files, keeping only the most recent `limit` jobs."""
    try:
        # Get all json job files sorted by modification time (newest first)
        job_files = sorted(
            JOBS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True
        )

        if len(job_files) <= limit:
            return

        # Delete jobs exceeding limit
        for job_file in job_files[limit:]:
            try:
                job_id = job_file.stem
                log_file = JOBS_DIR / f"{job_id}.log"

                # Delete json
                job_file.unlink()

                # Delete log if exists
                if log_file.exists():
                    log_file.unlink()

                logger.info(f"Cleaned up old job: {job_id}")
            except Exception as e:
                logger.error(f"Error deleting old job {job_file}: {e}")

    except Exception as e:
        logger.error(f"Error during job cleanup: {e}")


def update_job(job_id: str, **updates):
    """Update job status."""
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        return
    job_data = json.loads(job_file.read_text())
    job_data.update(updates)
    job_file.write_text(json.dumps(job_data, indent=2))


def get_job(job_id: str) -> Optional[dict]:
    """Get job data."""
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        return None
    return json.loads(job_file.read_text())


def is_any_job_running() -> bool:
    """Check if any job is currently running."""
    for job_file in JOBS_DIR.glob("*.json"):
        job_data = json.loads(job_file.read_text())
        if job_data.get("status") == "running":
            return True
    return False


def is_process_running(pid: int) -> bool:
    """Check if process with PID is still running."""
    try:
        # Try psutil first (more reliable)
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        # Fallback: use os.kill with signal 0
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


# ============================================================================
# TRAINING LOGIC
# ============================================================================


def check_promotion(stage: int, output_path: Path) -> tuple:
    """Always promote new model to production. Returns (promotion_dict, metrics_dict).

    No metric comparison - direct replace since training uses consistent
    parameters and 365-day backdate.
    """
    metadata_file = output_path / "training_metadata.json"

    # Load new metrics (for logging only)
    new_metrics = {}
    if metadata_file.exists():
        new_metadata = json.loads(metadata_file.read_text())
        new_metrics = new_metadata.get("metrics", {})

    # Extract metrics for logging
    if stage == 1:
        result_metrics = {
            "ndcg": round(new_metrics.get("val_ndcg", new_metrics.get("ndcg", 0)), 4),
        }
    else:  # stage 2
        result_metrics = {
            "mrr_combined": round(new_metrics.get("mrr_combined", 0), 4),
            "val_cosine": round(new_metrics.get("val_cosine", 0), 4),
        }

    # Always promote - no metric comparison needed
    promote_to_production(stage, output_path)
    return {
        "eligible": True,
        "reason": "Auto-promoted (no metric gate)",
        "promoted": True,
    }, result_metrics


def promote_to_production(stage: int, source_path: Path) -> bool:
    """Copy checkpoint to production folder with rollback on failure."""
    import shutil

    prod_path = CHECKPOINTS_DIR / f"production_stage{stage}"
    backup_path = CHECKPOINTS_DIR / f"production_stage{stage}_backup"

    try:
        # Backup current production
        if prod_path.exists():
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(str(prod_path), str(backup_path))
            logger.info(f"Backed up {prod_path} to {backup_path}")

        # Move new to production (Renames source to prod, eliminating redundancy)
        shutil.move(str(source_path), str(prod_path))
        logger.info(f"Promoted {source_path} to {prod_path}")
        return True

    except Exception as e:
        logger.error(f"Promotion failed: {e}")
        # Rollback: restore from backup
        if backup_path.exists() and not prod_path.exists():
            shutil.move(str(backup_path), str(prod_path))
            logger.info(f"Rolled back to previous version from {backup_path}")
        return False


def promote_both_stages_atomic(stage1_path: Path, stage2_path: Path) -> bool:
    """Promote BOTH stages atomically - all or nothing.

    If promotion fails for any reason, rollback BOTH stages to backup.
    Returns True on success, False on failure (with rollback applied).
    """
    import shutil

    prod1 = CHECKPOINTS_DIR / "production_stage1"
    prod2 = CHECKPOINTS_DIR / "production_stage2"
    backup1 = CHECKPOINTS_DIR / "production_stage1_backup"
    backup2 = CHECKPOINTS_DIR / "production_stage2_backup"

    try:
        # Step 1: Backup current production (both stages)
        logger.info("Backing up current production models...")
        for prod, backup in [(prod1, backup1), (prod2, backup2)]:
            if prod.exists():
                if backup.exists():
                    shutil.rmtree(backup)
                shutil.move(str(prod), str(backup))
                logger.info(f"Backed up {prod.name} to {backup.name}")

        # Step 2: Atomic move (both stages must succeed)
        logger.info("Promoting new models atomically...")
        shutil.move(str(stage1_path), str(prod1))
        logger.info(f"Promoted Stage 1 to {prod1}")
        shutil.move(str(stage2_path), str(prod2))
        logger.info(f"Promoted Stage 2 to {prod2}")

        logger.info("Atomic promotion complete: both stages updated")
        return True

    except Exception as e:
        logger.error(f"Atomic promotion failed: {e}")

        # Step 3: Rollback from backup (both stages)
        logger.info("Rolling back both stages to previous version...")
        for prod, backup in [(prod1, backup1), (prod2, backup2)]:
            if backup.exists():
                # Remove partial/new content if it exists
                if prod.exists():
                    shutil.rmtree(prod)
                # Restore from backup
                shutil.move(str(backup), str(prod))
                logger.info(f"Restored {prod.name} from {backup.name}")

        logger.info("Rollback complete")
        return False


def run_stage1(job_id: str, output_path: Path, start_date: str, end_date: str) -> dict:
    """Run Stage 1 training and return metrics.

    Returns dict with 'success' (bool) and 'metrics' (dict).
    """
    cmd = [
        "python",
        "-u",  # Unbuffered output for real-time logging
        "training/train_stage1.py",
        "--start_date",
        start_date,
        "--end_date",
        end_date,
        "--output_path",
        str(output_path),
        # Trial 25 Best Params (FIXED temperature)
        "--lr",
        "1.20e-05",
        "--temperature",
        "0.0869",  # REVERTED: 0.045 was too low, caused NDCG drop (0.775→0.648)
        "--triplet-weight",
        "2.862",
        "--class-weight",
        "0.3287",
        "--gate-l1-weight",
        "0.0038",
        "--suffix-drop-rate",
        "0.2492",
        "--category-dropout",
        "0.2994",
        "--warmup-ratio",
        "0.1561",
        "--epochs",
        "15",
    ]

    # Set PYTHONPATH for proper module imports
    project_root = Path(__file__).parent.absolute()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    # Log file path
    log_path = output_path / "training.log"

    logger.info(f"[{job_id}] Starting Stage 1 training: {cmd}")

    # Create log file BEFORE starting process (so it's immediately available)
    log_file = open(log_path, "w", buffering=1)

    # Run training with PIPE for tee
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(project_root),
        env=env,
    )

    # Register PID for graceful shutdown tracking
    ACTIVE_TRAINING_PIDS[job_id] = process.pid
    logger.info(f"[{job_id}] Registered PID {process.pid} for tracking")

    try:
        # Start tee thread (writes to file + console)
        # Must be non-daemon to ensure it completes before function returns
        tee_thread = threading.Thread(
            target=tee_output,
            args=(process, log_file, job_id, "stage1"),
            daemon=False,  # Changed: non-daemon to prevent premature death
        )
        tee_thread.start()

        # Wait for process completion
        process.wait()

        # Wait for tee thread to finish writing all output
        tee_thread.join(timeout=5.0)  # Wait up to 5 seconds for tee to finish
    finally:
        # Always close log file
        log_file.close()
        # Unregister PID when done (completed or failed)
        ACTIVE_TRAINING_PIDS.pop(job_id, None)
        logger.info(f"[{job_id}] Unregistered PID {process.pid}")

    if process.returncode != 0:
        error = f"Stage 1 training failed with exit code {process.returncode}"
        logger.error(f"[{job_id}] {error}")
        return {"success": False, "error": error}

    # Validate output
    if not (output_path / "best_model.pt").exists():
        error = "Stage 1 completed but no checkpoint found"
        logger.error(f"[{job_id}] {error}")
        return {"success": False, "error": error}

    # Load metrics
    metadata_file = output_path / "training_metadata.json"
    metrics = {}
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        metrics = metadata.get("metrics", {})

    logger.info(f"[{job_id}] Stage 1 complete: NDCG={metrics.get('ndcg', 'N/A')}")
    return {"success": True, "metrics": metrics}


def run_stage2(
    job_id: str, stage1_path: Path, output_path: Path, start_date: str, end_date: str
) -> dict:
    """Run Stage 2 training using Stage 1 outputs.

    Returns dict with 'success' (bool) and 'metrics' (dict).
    """
    cmd = [
        "python",
        "-u",  # Unbuffered output for real-time logging
        "training/train_stage2.py",
        "--epochs",
        "20",
        "--batch_size",
        "64",
        "--lr",
        "4.77e-05",
        "--max_length",
        "64",
        "--checkpoint_path",
        str(stage1_path),  # Uses Stage 1 outputs
        "--output_path",
        str(output_path),
        "--start_date",
        start_date,
        "--end_date",
        end_date,
        "--noise_prob",
        "0.05",
        "--warmup_ratio",
        "0.0529",
        "--loss_type",
        "hybrid",
        "--alignment_weight",
        "1.0",
        "--uniformity_weight",
        "0.1001",
        "--ranking_weight",
        "0.35",  # V5.4 standard
        "--hard_negative_weight",
        "0.1521",
        "--temperature",
        "0.1247",  # V5.4 standard
        "--synthetic_ratio",
        "0.35",  # V5.4 standard
        "--synthetic_data_path",
        "runtime/data/synthetic_merged.json",
    ]

    # Set PYTHONPATH for proper module imports
    project_root = Path(__file__).parent.absolute()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    # Log file path
    log_path = output_path / "training.log"

    logger.info(f"[{job_id}] Starting Stage 2 training: {cmd}")

    # Create log file BEFORE starting process (so it's immediately available)
    log_file = open(log_path, "w", buffering=1)

    # Run training with PIPE for tee
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(project_root),
        env=env,
    )

    # Register PID for graceful shutdown tracking
    ACTIVE_TRAINING_PIDS[job_id] = process.pid
    logger.info(f"[{job_id}] Registered PID {process.pid} for tracking")

    try:
        # Start tee thread (writes to file + console)
        # Must be non-daemon to ensure it completes before function returns
        tee_thread = threading.Thread(
            target=tee_output,
            args=(process, log_file, job_id, "stage2"),
            daemon=False,  # Changed: non-daemon to prevent premature death
        )
        tee_thread.start()

        # Wait for process completion
        process.wait()

        # Wait for tee thread to finish writing all output
        tee_thread.join(timeout=5.0)  # Wait up to 5 seconds for tee to finish
    finally:
        # Always close log file
        log_file.close()
        # Unregister PID when done (completed or failed)
        ACTIVE_TRAINING_PIDS.pop(job_id, None)
        logger.info(f"[{job_id}] Unregistered PID {process.pid}")

    if process.returncode != 0:
        error = f"Stage 2 training failed with exit code {process.returncode}"
        logger.error(f"[{job_id}] {error}")
        return {"success": False, "error": error}

    # Validate output
    if not (output_path / "enhanced_query_encoder.pt").exists():
        error = "Stage 2 completed but no checkpoint found"
        logger.error(f"[{job_id}] {error}")
        return {"success": False, "error": error}

    # Load metrics
    metadata_file = output_path / "training_metadata.json"
    metrics = {}
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        metrics = metadata.get("metrics", {})

    logger.info(
        f"[{job_id}] Stage 2 complete: MRR={metrics.get('mrr_combined', 'N/A')}"
    )
    return {"success": True, "metrics": metrics}


def run_unified_training(job_id: str, start_date: str, end_date: str):
    """Run Stage 1 + Stage 2 as atomic pipeline.

    Flow:
    1. Create output folder: full_<timestamp>/stage1/ and stage2/
    2. Run Stage 1 -> validate checkpoint
    3. Run Stage 2 -> validate checkpoint
    4. Atomic promotion (both or rollback)
    5. If ANY step fails: DELETE all outputs (atomic cleanup)
    """
    from pathlib import Path
    import shutil
    import time

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_base = CHECKPOINTS_DIR / f"full_{timestamp}"
    stage1_path = output_base / "stage1"
    stage2_path = output_base / "stage2"

    start_time = time.time()
    metrics = {"stage1": {}, "stage2": {}}

    try:
        update_job(job_id, status="running", started_at=datetime.now().isoformat())
        logger.info(f"[{job_id}] Unified training started")

        # === STAGE 1 ===
        logger.info(f"[{job_id}] Starting Stage 1...")
        update_job(job_id, substage="stage1")

        stage1_path.mkdir(parents=True, exist_ok=True)
        result1 = run_stage1(job_id, stage1_path, start_date, end_date)

        if not result1["success"]:
            raise RuntimeError(
                f"Stage 1 failed: {result1.get('error', 'Unknown error')}"
            )

        metrics["stage1"] = result1["metrics"]
        logger.info(f"[{job_id}] Stage 1 complete")

        # === STAGE 2 ===
        logger.info(f"[{job_id}] Starting Stage 2...")
        update_job(job_id, substage="stage2")

        stage2_path.mkdir(parents=True, exist_ok=True)
        result2 = run_stage2(job_id, stage1_path, stage2_path, start_date, end_date)

        if not result2["success"]:
            raise RuntimeError(
                f"Stage 2 failed: {result2.get('error', 'Unknown error')}"
            )

        metrics["stage2"] = result2["metrics"]
        logger.info(f"[{job_id}] Stage 2 complete")

        # === ATOMIC PROMOTION ===
        logger.info(f"[{job_id}] Promoting both stages atomically...")
        if not promote_both_stages_atomic(stage1_path, stage2_path):
            raise RuntimeError("Atomic promotion failed")

        # === SUCCESS ===
        duration = (time.time() - start_time) / 60
        update_job(
            job_id,
            status="completed",
            substage=None,
            completed_at=datetime.now().isoformat(),
            duration_minutes=round(duration, 1),
            metrics=metrics,
            promotion={
                "promoted": True,
                "note": "Atomic: both stages updated together",
            },
            output_path=str(output_base),
        )
        logger.info(f"[{job_id}] Job completed successfully in {duration:.1f} minutes")

    except Exception as e:
        # === ATOMIC CLEANUP ===
        logger.error(f"[{job_id}] Failed: {e}")

        # Delete all outputs (atomic all-or-nothing)
        if output_base.exists():
            logger.info(f"[{job_id}] Cleaning up: {output_base}")
            shutil.rmtree(output_base)

        # Mark job as failed
        duration = (time.time() - start_time) / 60
        update_job(
            job_id,
            status="failed",
            substage=None,
            error=str(e),
            completed_at=datetime.now().isoformat(),
            duration_minutes=round(duration, 1),
            promotion={"promoted": False, "note": "Failed - no promotion"},
            output_path=str(output_base),
        )
        logger.error(f"[{job_id}] Job failed after {duration:.1f} minutes")


# ============================================================================
# APP SETUP
# ============================================================================


def clean_zombie_jobs():
    """Mark any 'running' jobs as 'failed' on startup."""
    count = 0
    for job_file in JOBS_DIR.glob("*.json"):
        try:
            job_data = json.loads(job_file.read_text())
            if job_data.get("status") == "running":
                logger.warning(f"Found zombie job {job_file.name}. Marking as failed.")
                update_job(
                    job_data["job_id"],
                    status="failed",
                    error="System restarted while job was running (Zombie Job)",
                )
                count += 1
        except Exception as e:
            logger.error(f"Error checking job file {job_file}: {e}")
    if count > 0:
        logger.info(f"Cleaned up {count} zombie jobs.")


def check_promotion_for_full(stage: str, output_path: Path) -> tuple:
    """Check promotion for full pipeline (both stages)."""
    stage1_path = output_path / "stage1"
    stage2_path = output_path / "stage2"

    # Load metrics from both stages
    stage1_metadata = stage1_path / "training_metadata.json"
    stage2_metadata = stage2_path / "training_metadata.json"

    metrics = {"stage1": {}, "stage2": {}}

    if stage1_metadata.exists():
        metadata = json.loads(stage1_metadata.read_text())
        metrics["stage1"] = {
            "ndcg": round(metadata.get("metrics", {}).get("ndcg", 0), 4)
        }

    if stage2_metadata.exists():
        metadata = json.loads(stage2_metadata.read_text())
        stage2_metrics = metadata.get("metrics", {})
        metrics["stage2"] = {
            "mrr_combined": round(stage2_metrics.get("mrr_combined", 0), 4),
            "val_cosine": round(stage2_metrics.get("val_cosine", 0), 4),
        }

    # Atomic promote (already done by run_unified_training, just confirm)
    return {
        "eligible": True,
        "reason": "Auto-promoted (atomic pipeline)",
        "promoted": True,
    }, metrics


async def monitor_jobs():
    """Background task to monitor running jobs and update their status."""
    while True:
        try:
            for job_file in JOBS_DIR.glob("*.json"):
                try:
                    job_data = json.loads(job_file.read_text())
                    job_id = job_data.get("job_id")
                    status = job_data.get("status")
                    pid = job_data.get("pid")

                    if status == "running" and pid:
                        # Check if process still running
                        if not is_process_running(pid):
                            # Process finished - check exit
                            output_path = job_data.get("output_path")
                            stage = job_data.get("stage")

                            # Check completion based on stage type
                            if stage == "full":
                                # Unified pipeline: check for final completion
                                # Look for completion marker or check both stage files
                                stage1_path = (
                                    Path(output_path) / "stage1"
                                    if output_path
                                    else None
                                )
                                stage2_path = (
                                    Path(output_path) / "stage2"
                                    if output_path
                                    else None
                                )

                                if stage1_path and stage2_path:
                                    stage1_ok = (stage1_path / "best_model.pt").exists()
                                    stage2_ok = (
                                        stage2_path / "enhanced_query_encoder.pt"
                                    ).exists()

                                    if stage1_ok and stage2_ok:
                                        # Both complete - promote and mark done
                                        try:
                                            promotion, metrics = (
                                                check_promotion_for_full(
                                                    stage, Path(output_path)
                                                )
                                            )
                                            started = datetime.fromisoformat(
                                                job_data["started_at"]
                                            )
                                            duration = (
                                                datetime.now() - started
                                            ).total_seconds() / 60

                                            update_job(
                                                job_id,
                                                status="completed",
                                                substage=None,  # Clear substage
                                                completed_at=datetime.now().isoformat(),
                                                duration_minutes=round(duration, 1),
                                                metrics=metrics,
                                                promotion=promotion,
                                            )
                                            logger.info(
                                                f"[{job_id}] Completed. Promotion: {promotion}"
                                            )
                                        except Exception as e:
                                            update_job(
                                                job_id,
                                                status="failed",
                                                substage=None,
                                                error=f"Post-processing failed: {str(e)}",
                                                completed_at=datetime.now().isoformat(),
                                            )
                                    else:
                                        # Incomplete - failed
                                        substage_info = (
                                            f" (stopped at {job_data.get('substage', 'unknown')})"
                                            if job_data.get("substage")
                                            else ""
                                        )
                                        update_job(
                                            job_id,
                                            status="failed",
                                            substage=None,
                                            error=f"Pipeline incomplete{substage_info}",
                                            completed_at=datetime.now().isoformat(),
                                        )
                                        logger.error(f"[{job_id}] Pipeline incomplete")
                            else:
                                # Legacy stage 1 or 2
                                model_file = (
                                    "best_model.pt"
                                    if stage == 1
                                    else "enhanced_query_encoder.pt"
                                )
                                if (
                                    output_path
                                    and (Path(output_path) / model_file).exists()
                                ):
                                    try:
                                        promotion, metrics = check_promotion(
                                            stage, Path(output_path)
                                        )
                                        started = datetime.fromisoformat(
                                            job_data["started_at"]
                                        )
                                        duration = (
                                            datetime.now() - started
                                        ).total_seconds() / 60

                                        update_job(
                                            job_id,
                                            status="completed",
                                            completed_at=datetime.now().isoformat(),
                                            duration_minutes=round(duration, 1),
                                            metrics=metrics,
                                            promotion=promotion,
                                        )
                                        logger.info(
                                            f"[{job_id}] Completed. Promotion: {promotion}"
                                        )
                                    except Exception as e:
                                        update_job(
                                            job_id,
                                            status="failed",
                                            error=f"Post-processing failed: {str(e)}",
                                            completed_at=datetime.now().isoformat(),
                                        )
                                else:
                                    update_job(
                                        job_id,
                                        status="failed",
                                        error="Process died unexpectedly (no output found)",
                                        completed_at=datetime.now().isoformat(),
                                    )
                                    logger.error(
                                        f"[{job_id}] Process died unexpectedly"
                                    )

                except Exception as e:
                    logger.error(f"Error monitoring job {job_file}: {e}")

            await asyncio.sleep(10)  # Check every 10 seconds

        except Exception as e:
            logger.error(f"Error in monitor_jobs: {e}")
            await asyncio.sleep(10)


def cleanup_active_processes():
    """Kill all active training processes on shutdown.

    Prevents orphan processes when uvicorn is stopped.
    """
    if not ACTIVE_TRAINING_PIDS:
        logger.info("No active training processes to cleanup")
        return

    logger.warning(
        f"Cleaning up {len(ACTIVE_TRAINING_PIDS)} active training processes..."
    )
    for job_id, pid in list(ACTIVE_TRAINING_PIDS.items()):
        try:
            if is_process_running(pid):
                logger.info(f"[{job_id}] Killing PID {pid}...")
                # Try graceful shutdown first
                import signal

                try:
                    os.kill(pid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass

                # Force kill if still alive after 2 seconds
                import time

                time.sleep(2)
                if is_process_running(pid):
                    logger.warning(f"[{job_id}] Force killing PID {pid}...")
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass

                # Mark job as failed
                update_job(
                    job_id,
                    status="failed",
                    error="Training killed during server shutdown",
                    completed_at=datetime.now().isoformat(),
                )
            else:
                logger.info(f"[{job_id}] PID {pid} already stopped")
        except Exception as e:
            logger.error(f"[{job_id}] Error killing PID {pid}: {e}")

    ACTIVE_TRAINING_PIDS.clear()
    logger.info("Cleanup complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Clean up zombie jobs
    clean_zombie_jobs()

    # Start background job monitor
    monitor_task = asyncio.create_task(monitor_jobs())
    logger.info("Background job monitor started")

    yield

    # Shutdown: Kill all active training processes
    logger.info("Shutting down - cleaning up active training processes...")
    cleanup_active_processes()

    # Cancel monitor task
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
    logger.info("Background job monitor stopped")


app = FastAPI(title="RetailCo Training API", lifespan=lifespan)


@app.get("/health")
def health_check():
    """Health check."""
    return {"status": "ok", "jobs_dir": str(JOBS_DIR)}


@app.post("/train/full", response_model=JobResponse)
async def train_full(request: TrainRequest):
    """Trigger unified Stage 1 + Stage 2 training.

    Runs both stages sequentially in atomic pipeline.
    Only promotes if BOTH stages succeed.
    """
    if is_any_job_running():
        raise HTTPException(status_code=409, detail="Another job is running")

    # Validate date order
    if request.start_date > request.end_date:
        logger.warning(
            f"Date order warning: start_date ({request.start_date}) > end_date ({request.end_date})"
        )
        raise HTTPException(
            status_code=400, detail="start_date must be before or equal to end_date"
        )

    job_id = create_job("full", request.start_date, request.end_date)

    # Run in background (fire and forget)
    import threading

    thread = threading.Thread(
        target=run_unified_training,
        args=(job_id, request.start_date, request.end_date),
        daemon=False,  # Changed: non-daemon to prevent tee threads from dying
    )
    thread.start()

    return JobResponse(
        job_id=job_id, status="queued", message="Unified training pipeline started"
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get job status with real-time PID checking."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Real-time check: if status is "running", verify process is actually running
    if job.get("status") == "running" and "pid" in job:
        if not is_process_running(job["pid"]):
            # Process died - update status immediately
            update_job(
                job_id,
                status="failed",
                error="Process died unexpectedly (detected on status check)",
                completed_at=datetime.now().isoformat(),
            )
            job = get_job(job_id)  # Reload updated job

    return JobStatus(**job)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_training:app", host="0.0.0.0", port=8001, reload=True)
