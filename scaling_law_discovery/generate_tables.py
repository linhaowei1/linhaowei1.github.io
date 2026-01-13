#!/usr/bin/env python3
"""
Generate LaTeX tables from JSONL data files.
Follows the same logic as index.html for loading and processing data.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# Task mapping
TASK_MAPPING = {
    "parallel_scaling_law": "parallel",
    "vocab_scaling_law": "vocab_size",
    "sft_scaling_law": "SFT",
    "domain_mixture_scaling_law": "domain_mix",
    "moe_scaling_law": "moe",
    "data_constrained_scaling_law": "d_constrain",
    "lr_bsz_scaling_law": "lr&bsz",
    "easy_question_scaling_law": "u_shape",
}

TASK_ORDER = [
    "parallel_scaling_law",
    "vocab_scaling_law",
    "sft_scaling_law",
    "domain_mixture_scaling_law",
    "moe_scaling_law",
    "data_constrained_scaling_law",
    "lr_bsz_scaling_law",
    "easy_question_scaling_law",
]

# Agent display names
AGENT_DISPLAY_NAMES = {
    "aider": "Aider",
    "terminus-2": "Terminus-2",
    "mini-swe-agent": "Mini-SWE-Agent",
    "opencode": "OpenCode",
    "openhands": "OpenHands",
    "codex": "CodeX",
    "goose": "Goose",
    "sldagent": "SLDAgent",
    "claude-code": "Claude Code",
    "gemini-cli": "Gemini-CLI",
}

# Model display names
MODEL_DISPLAY_NAMES = {
    "gpt-5": "GPT-5",
    "gpt-5.2": "GPT-5.2",
    "gpt-4.1": "GPT-4.1",
    "gpt-4o": "GPT-4o",
    "o1": "o1",
    "o3": "o3",
    "o4-mini": "o4-mini",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "DeepSeek-V3.2": "DeepSeek V3.2",
    "DeepSeek-V3.2-reasoning": "DeepSeek V3.2-R",
}


def load_jsonl(filepath):
    """Load JSONL file and return list of entries."""
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def normalize_model_name(model):
    """Normalize model names to handle variations."""
    model_lower = model.lower()
    # Handle Claude variations
    if "claude-haiku-4-5" in model_lower or "claude-haiku-4.5" in model_lower:
        return "claude-haiku-4-5"
    if "claude-sonnet-4-5" in model_lower or "claude-sonnet-4.5" in model_lower:
        return "claude-sonnet-4-5"
    # Handle Gemini variations
    if "gemini-2.5-flash" in model_lower or "gemini-2-5-flash" in model_lower:
        return "gemini-2.5-flash"
    if "gemini-2.5-pro" in model_lower or "gemini-2-5-pro" in model_lower:
        return "gemini-2.5-pro"
    if "gemini-3-pro" in model_lower:
        return "gemini-3-pro-preview"
    if "gemini-3-flash" in model_lower:
        return "gemini-3-flash-preview"
    return model_lower


def process_data(data_dir):
    """Process all JSONL files and compute statistics."""
    data_dir = Path(data_dir)
    
    # Group data by agent+model+task
    groups = defaultdict(lambda: {"runs": []})
    
    # Load all task files
    for task in TASK_ORDER:
        filepath = data_dir / f"{task}.jsonl"
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue
        
        entries = load_jsonl(filepath)
        
        for entry in entries:
            agent = entry.get("agent_name", "")
            model = entry.get("model_name", "")
            r2 = entry.get("reward_r2")
            
            if r2 is None:
                continue
            
            # Normalize model name
            model = normalize_model_name(model)
            # Keep agent name as-is (case-sensitive)
            agent_key = agent.lower()  # Use lowercase for key matching
            
            key = (agent_key, model, task)
            groups[key]["runs"].append(r2)
    
    # Process each group: pad to 5 runs, calculate mean
    results = {}
    for (agent, model, task), group_data in groups.items():
        runs = group_data["runs"]
        
        # If no runs at all, skip (will show as NA in table)
        if len(runs) == 0:
            continue
        
        # Pad to 5 runs with -1.0 if needed
        while len(runs) < 5:
            runs.append(-1.0)
        
        # Calculate mean including -1.0 values (they are valid)
        mean_r2 = sum(runs) / len(runs)
        
        results[(agent, model, task)] = {
            "mean_r2": mean_r2,
            "runs": runs,
        }
    
    return results


def format_r2(r2):
    """Format R² value to 3 decimal places."""
    if r2 is None:
        return "NA"
    return f"{r2:.3f}"


def generate_table1(results):
    """Generate Table 1: GPT-5 agents."""
    # Filter for GPT-5 only
    gpt5_results = {
        k: v for k, v in results.items()
        if k[1] == "gpt-5"
    }
    
    # Get all agents for GPT-5
    agents = set(k[0] for k in gpt5_results.keys())
    
    # Build task scores for each agent
    agent_scores = {}
    for agent_key in agents:
        agent_scores[agent_key] = {}
        for task in TASK_ORDER:
            key = (agent_key, "gpt-5", task)
            if key in gpt5_results:
                agent_scores[agent_key][task] = gpt5_results[key]["mean_r2"]
            else:
                agent_scores[agent_key][task] = None  # No data available
    
    # Calculate averages (include -1.0 as valid value, exclude None)
    for agent in agent_scores:
        scores = [v for v in agent_scores[agent].values() if v is not None]
        if scores:
            agent_scores[agent]["avg"] = sum(scores) / len(scores)
        else:
            agent_scores[agent]["avg"] = None
    
    # Add Human baseline (from the original table)
    agent_scores["human"] = {
        "parallel_scaling_law": 1.000,
        "vocab_scaling_law": 0.966,
        "sft_scaling_law": 0.957,
        "domain_mixture_scaling_law": 0.671,
        "moe_scaling_law": 0.703,
        "data_constrained_scaling_law": 0.911,
        "lr_bsz_scaling_law": -0.076,
        "easy_question_scaling_law": -1.000,
        "avg": 0.517,
    }
    
    # Sort agents by average (descending), but put SLDAgent and Human at specific positions
    agent_order = [
        "aider",
        "terminus-2",
        "mini-swe-agent",
        "opencode",
        "openhands",
        "codex",
        "goose",
        "sldagent",  # lowercase key
        "human",
    ]
    
    # Find best and second-best for each task
    task_best = {}
    task_second = {}
    
    for task in TASK_ORDER:
        task_scores = []
        for agent in agent_order:
            if agent in agent_scores and task in agent_scores[agent]:
                score = agent_scores[agent][task]
                if score is not None:  # Include -1.0 as valid value
                    task_scores.append((agent, score))
        
        if task_scores:
            task_scores.sort(key=lambda x: x[1], reverse=True)
            if len(task_scores) >= 1:
                task_best[task] = task_scores[0][0]
            if len(task_scores) >= 2:
                task_second[task] = task_scores[1][0]
    
    # Generate LaTeX
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("")
    lines.append("\\centering")
    lines.append("")
    lines.append("\\vspace{-0.2em}")
    lines.append("")
    lines.append("\\caption{Performance on SLDBench for agents using GPT-5. Scores are $R^2$ averaged over five runs. The \\textbf{best} and \\underline{second-best} scores for each task are highlighted. ``NA'' indicates no valid output.}")
    lines.append("")
    lines.append("\\label{tab:gpt5}")
    lines.append("")
    lines.append("\\Huge")
    lines.append("")
    lines.append("\\renewcommand{\\arraystretch}{1.2}")
    lines.append("")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("")
    lines.append("\\begin{tabular}{lccccccccc}")
    lines.append("")
    lines.append("\\toprule")
    lines.append("")
    
    # Header
    header = "& "
    task_headers = []
    for t in TASK_ORDER:
        task_name = TASK_MAPPING[t]
        # Escape & in LaTeX
        if "&" in task_name:
            task_name = task_name.replace("&", "\\&")
        task_headers.append(f"\\texttt{{{task_name}}}$\\uparrow$")
    header += " & ".join(task_headers)
    header += " & \\textbf{Avg. R$^2$}$\\uparrow$ \\\\"
    lines.append(header)
    lines.append("")
    lines.append("\\midrule")
    lines.append("")
    
    # Rows
    for agent_key in agent_order:
        if agent_key not in agent_scores:
            continue
        
        agent_display = AGENT_DISPLAY_NAMES.get(agent_key, agent_key.title())
        row = agent_display
        
        for task in TASK_ORDER:
            score = agent_scores[agent_key][task]
            formatted = format_r2(score)
            
            # Apply formatting
            if agent_key == task_best.get(task):
                formatted = f"\\textbf{{{formatted}}}"
            elif agent_key == task_second.get(task):
                formatted = f"\\underline{{{formatted}}}"
            
            row += f" & {formatted}"
        
        # Average
        avg_score = agent_scores[agent_key]["avg"]
        avg_formatted = format_r2(avg_score)
        if agent_key == "sldagent":
            avg_formatted = f"\\textbf{{{avg_formatted}}}"
        elif agent_key == "goose":
            avg_formatted = f"\\underline{{{avg_formatted}}}"
        row += f" & {avg_formatted} \\\\"
        
        lines.append(row)
        lines.append("")
    
    lines.append("\\bottomrule")
    lines.append("")
    lines.append("\\end{tabular}%")
    lines.append("")
    lines.append("}")
    lines.append("")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_table2(results):
    """Generate Table 2: Provider-specific agents."""
    # Define the agent-model pairs for this table
    pairs = [
        ("gemini-cli", "gemini-2.5-flash", "Gemini-CLI (G-2.5-Flash)"),
        ("sldagent", "gemini-2.5-flash", "SLDAgent (G-2.5-Flash)"),
        ("gemini-cli", "gemini-3-pro-preview", "Gemini-CLI (G-3-Pro)"),
        ("sldagent", "gemini-3-pro-preview", "SLDAgent (G-3-Pro)"),
        ("claude-code", "claude-haiku-4-5", "Claude Code (C-Haiku-4.5)"),
        ("sldagent", "claude-haiku-4-5", "SLDAgent (C-Haiku-4.5)"),
        ("claude-code", "claude-sonnet-4-5", "Claude Code (C-Sonnet-4.5)"),
        ("sldagent", "claude-sonnet-4-5", "SLDAgent (C-Sonnet-4.5)"),
        ("codex", "o4-mini", "CodeX (o4-mini)"),
        ("sldagent", "o4-mini", "SLDAgent (o4-mini)"),
        ("codex", "gpt-5", "CodeX (GPT-5)"),
        ("sldagent", "gpt-5", "SLDAgent (GPT-5)"),
        ("human", None, "Human"),
    ]
    
    # Normalize agent keys to lowercase for lookup
    agent_key_map = {
        "gemini-cli": "gemini-cli",
        "sldagent": "sldagent",
        "claude-code": "claude-code",
        "codex": "codex",
        "human": "human",
    }
    
    # Build scores
    pair_scores = {}
    for agent, model, display_name in pairs:
        pair_scores[display_name] = {}
        for task in TASK_ORDER:
            if model is None:  # Human baseline
                # Use same values as Table 1
                if task == "parallel_scaling_law":
                    pair_scores[display_name][task] = 1.000
                elif task == "vocab_scaling_law":
                    pair_scores[display_name][task] = 0.966
                elif task == "sft_scaling_law":
                    pair_scores[display_name][task] = 0.957
                elif task == "domain_mixture_scaling_law":
                    pair_scores[display_name][task] = 0.671
                elif task == "moe_scaling_law":
                    pair_scores[display_name][task] = 0.703
                elif task == "data_constrained_scaling_law":
                    pair_scores[display_name][task] = 0.911
                elif task == "lr_bsz_scaling_law":
                    pair_scores[display_name][task] = -0.076
                elif task == "easy_question_scaling_law":
                    pair_scores[display_name][task] = -1.000
            else:
                agent_key = agent_key_map.get(agent, agent.lower())
                key = (agent_key, model, task)
                if key in results:
                    pair_scores[display_name][task] = results[key]["mean_r2"]
                else:
                    pair_scores[display_name][task] = None  # No data available
        
        # Calculate average (include -1.0 as valid value, exclude None)
        scores = [v for v in pair_scores[display_name].values() if v is not None]
        if scores:
            pair_scores[display_name]["avg"] = sum(scores) / len(scores)
        else:
            pair_scores[display_name]["avg"] = None
    
    # Fix Human baseline average (should be 0.517, as specified in the original table)
    if "Human" in pair_scores:
        pair_scores["Human"]["avg"] = 0.517
    
    # Find best within each model family
    model_families = {
        "G-2.5-Flash": ["Gemini-CLI (G-2.5-Flash)", "SLDAgent (G-2.5-Flash)"],
        "G-3-Pro": ["Gemini-CLI (G-3-Pro)", "SLDAgent (G-3-Pro)"],
        "C-Haiku-4.5": ["Claude Code (C-Haiku-4.5)", "SLDAgent (C-Haiku-4.5)"],
        "C-Sonnet-4.5": ["Claude Code (C-Sonnet-4.5)", "SLDAgent (C-Sonnet-4.5)"],
        "o4-mini": ["CodeX (o4-mini)", "SLDAgent (o4-mini)"],
        "GPT-5": ["CodeX (GPT-5)", "SLDAgent (GPT-5)"],
    }
    
    task_best_by_family = {}
    for family, pair_names in model_families.items():
        task_best_by_family[family] = {}
        for task in TASK_ORDER:
            best_score = -999
            best_name = None
            for name in pair_names:
                if name in pair_scores and task in pair_scores[name]:
                    score = pair_scores[name][task]
                    if score is not None and score > best_score:  # Include -1.0 as valid value
                        best_score = score
                        best_name = name
            if best_name:
                task_best_by_family[family][task] = best_name
    
    # Generate LaTeX
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("")
    lines.append("\\centering")
    lines.append("")
    lines.append("\\vspace{-0.2em}")
    lines.append("")
    lines.append("\\caption{SLDBench performance of provider-specific agents together with reference rows for {SLDAgent} paired with the corresponding provider models. ``G-2.5-Flash'' denotes Gemini-2.5-Flash, ``G-3-Pro'' denotes Gemini-3-Pro-Preview, ``C-Haiku-4.5'' denotes Claude-Haiku-4.5, and ``C-Sonnet-4.5'' denotes Claude-Sonnet-4.5. Scores report $R^2$, averaged over five runs. Bold denotes the best value within each model family.}")
    lines.append("")
    lines.append("\\label{tab:non_o4}")
    lines.append("")
    lines.append("\\Huge")
    lines.append("")
    lines.append("\\renewcommand{\\arraystretch}{1.2}")
    lines.append("")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("")
    lines.append("\\begin{tabular}{lccccccccc}")
    lines.append("")
    lines.append("\\toprule")
    lines.append("")
    
    # Header
    header = "& "
    task_headers = []
    for t in TASK_ORDER:
        task_name = TASK_MAPPING[t]
        # Escape & in LaTeX
        if "&" in task_name:
            task_name = task_name.replace("&", "\\&")
        task_headers.append(f"\\texttt{{{task_name}}}$\\uparrow$")
    header += " & ".join(task_headers)
    header += " & \\textbf{Avg. R$^2$}$\\uparrow$ \\\\"
    lines.append(header)
    lines.append("")
    lines.append("\\midrule")
    lines.append("")
    
    # Rows
    for agent, model, display_name in pairs:
        row = display_name
        
        # Determine which family this belongs to
        family = None
        for f, names in model_families.items():
            if display_name in names:
                family = f
                break
        
        for task in TASK_ORDER:
            score = pair_scores[display_name][task]
            formatted = format_r2(score)
            
            # Apply bold if best in family
            if family and task_best_by_family.get(family, {}).get(task) == display_name:
                formatted = f"\\textbf{{{formatted}}}"
            
            row += f" & {formatted}"
        
        # Average
        avg_score = pair_scores[display_name]["avg"]
        avg_formatted = format_r2(avg_score)
        if family and display_name.startswith("SLDAgent"):
            avg_formatted = f"\\textbf{{{avg_formatted}}}"
        row += f" & {avg_formatted} \\\\"
        
        lines.append(row)
        lines.append("")
        
        # Add midrule after each pair (except last)
        if display_name != "Human":
            next_idx = pairs.index((agent, model, display_name)) + 1
            if next_idx < len(pairs):
                next_pair = pairs[next_idx]
                # Check if next pair starts a new family
                if next_pair[2].startswith("Gemini-CLI") or next_pair[2].startswith("Claude Code") or next_pair[2].startswith("CodeX"):
                    lines.append("\\midrule")
                    lines.append("")
    
    lines.append("\\bottomrule")
    lines.append("")
    lines.append("\\end{tabular}%")
    lines.append("")
    lines.append("}")
    lines.append("")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def main():
    data_dir = Path(__file__).parent / "data"
    
    print("Loading data...")
    results = process_data(data_dir)
    print(f"Processed {len(results)} agent-model-task combinations")
    
    print("\nGenerating Table 1...")
    table1 = generate_table1(results)
    
    print("\nGenerating Table 2...")
    table2 = generate_table2(results)
    
    # Write output
    output_file = Path(__file__).parent / "tables.tex"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("% Table 1: GPT-5 agents\n")
        f.write(table1)
        f.write("\n\n")
        f.write("% Table 2: Provider-specific agents\n")
        f.write(table2)
    
    print(f"\nTables written to {output_file}")
    print("\n" + "="*80)
    print("TABLE 1:")
    print("="*80)
    print(table1)
    print("\n" + "="*80)
    print("TABLE 2:")
    print("="*80)
    print(table2)


if __name__ == "__main__":
    main()

