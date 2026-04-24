import re
import pandas as pd
import wandb


def _section_to_prefix(section: str | None) -> str | None:
    if not section:
        return None
    if section.startswith("training"):
        return "train"
    if section.startswith("time"):
        return "time"
    if section.startswith("environment"):
        return "env"
    if section.startswith("evaluation") or section.startswith("eval"):
        return "eval"
    return None


def _detect_section(line: str, current_section: str | None) -> str | None:
    lower_line = line.lower()

    # New metric block starts
    if "metric table" in lower_line:
        return None

    # Section headers in RLinf table
    if "training/actor" in lower_line or "training/critic" in lower_line or "training/other" in lower_line:
        return "training"
    if "environment" in lower_line and ("├" in line or "┤" in line):
        return "environment"
    if ("evaluation" in lower_line or " eval " in f" {lower_line} ") and ("├" in line or "┤" in line):
        return "evaluation"
    if " time " in f" {lower_line} " and ("├" in line or "┤" in line):
        return "time"

    return current_section


def parse_slurm_log(file_path):
    all_data = []
    current_metrics = {}
    current_section = None
    
    step_pattern = re.compile(r"Global Step:\s+(\d+)")
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
    # Nur echte Metrikpaare wie actor/lr=1.00e-04
    # Key muss mit Buchstaben starten, damit wir Log-Artefakte ausschliessen.
    metric_pattern = re.compile(
        r"([A-Za-z][\w\/\-\_]*)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    with open(file_path, "r") as f:
        for line in f:
            # Farbcodes entfernen und Box-Zeichen aufräumen
            line = ansi_pattern.sub("", line)
            clean_line = line.replace("│", " ")

            # Step finden
            step_match = step_pattern.search(clean_line)
            if step_match:
                if current_metrics:
                    all_data.append(current_metrics)
                current_metrics = {"global_step": int(step_match.group(1))}
                current_section = None
            
            # Aktuelle Tabelle-Sektion mitführen
            current_section = _detect_section(line, current_section)

            # Nur Zeilen aus der ASCII-Tabelle als Metrikzeilen behandeln.
            if "│" not in line:
                continue

            # Alle Metriken in der Zeile finden
            matches = metric_pattern.findall(clean_line)
            
            for key, value in matches:
                clean_key = key.strip()
                metric_value = float(value)
                current_metrics[clean_key] = metric_value

                # Section-basierte Prefixe für gezielten Export
                section_prefix = _section_to_prefix(current_section)
                if section_prefix:
                    if clean_key.startswith(f"{section_prefix}/"):
                        prefixed_key = clean_key
                    else:
                        prefixed_key = f"{section_prefix}/{clean_key}"
                    current_metrics[prefixed_key] = metric_value
        
        if current_metrics:
            all_data.append(current_metrics)
            
    df = pd.DataFrame(all_data)
    allowed_prefixes = ("train/", "time/", "eval/", "env/")
    keep_cols = ["global_step"] + [c for c in df.columns if c.startswith(allowed_prefixes)]

    # Unerwuenschte Felder entfernen
    drop_tokens = ("/rank", "/pid")
    keep_cols = [c for c in keep_cols if c == "global_step" or not any(t in c for t in drop_tokens)]
    return df[keep_cols]

# --- Anwendung ---
df = parse_slurm_log("slurm-3967858.out")
df.to_csv("metrics_cleaned.csv", index=False)

# Optional: Zurück zu W&B hochladen (nur train/time/eval/env)
UPLOAD_TO_WANDB = True
WANDB_GROUP = 'DSRL'
WANDB_PROJECT = "rlinf"
WANDB_RUN_NAME = "metaworld_50_task2trial0_pi0_trafocritic_repeatPolicy_restored"

if UPLOAD_TO_WANDB:
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, group=WANDB_GROUP)
    for _, row in df.iterrows():
        clean_row = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
        # Manche Zeilen enthalten keine gueltige Step-Info -> ueberspringen
        if "global_step" not in clean_row:
            continue
        step = int(clean_row.pop("global_step"))
        if not clean_row:
            continue
        wandb.log(clean_row, step=step)
    wandb.finish()

print(f"Fertig! {len(df)} Datenpunkte wiederhergestellt.")
