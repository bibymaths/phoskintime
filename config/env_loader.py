import os

from dotenv import load_dotenv


def load_env_variables(env_path=".env"):
    load_dotenv(dotenv_path=env_path)
    return {
        "OUTPUT_DIR": os.getenv("OUTPUT_DIR", "distributive_profiles"),
        "DATA_FILE": os.getenv("DATA_FILE", "optimization_results.xlsx"),
        "SHEET_NAME": os.getenv("SHEET_NAME", "Estimated Values")
    }


# 3. Sample .env file (to place in project root)
# OUTPUT_DIR=distributive_profiles
# DATA_FILE=optimization/optimization_results.xlsx
# SHEET_NAME=Estimated Values