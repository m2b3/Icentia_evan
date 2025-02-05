#!/bin/bash
#SBATCH --job-name=heart_rate generation (neruokit)
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=05:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --account=rrg-skrishna  # Your project account

# Load Python
module load python/3.8  # Ensure the correct Python version

# Run the script
python /home/evan1/projects/rrg-skrishna/evan1/Ecg_rate_neurokit.py

