#!/bin/bash

# The full storage URL
export OPTUNA_STORAGE="sqlite:///${OPTUNA_DB_PATH}"

echo "Creating Optuna database at: $OPTUNA_STORAGE"

# Initialize the database
python << EOF
import optuna

storage = "$OPTUNA_STORAGE"

# Create a test study to initialize the database
study = optuna.create_study(
    study_name="initialization_test",
    storage=storage,
    direction="minimize",
)

print(f"✓ Database initialized successfully")
print(f"✓ Storage URL: {storage}")

# Clean up test study
optuna.delete_study(study_name="initialization_test", storage=storage)
EOF

echo ""
echo "Setup complete! Use this storage URL in your scripts:"
echo "STORAGE=\"$OPTUNA_STORAGE\""
