#!/bin/sh
set -e

# This script is run *after* the primary POSTGRES_DB is created.
# We connect to the primary DB to run our SQL command.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create the new database for MLflow
    CREATE DATABASE $MLFLOW_DB;
    
    -- Grant all privileges on the new database to our user
    GRANT ALL PRIVILEGES ON DATABASE $MLFLOW_DB TO $POSTGRES_USER;
EOSQL