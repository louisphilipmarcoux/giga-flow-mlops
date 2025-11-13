# **Giga-Flow MLOps 🚀**

This project is a complete, end-to-end MLOps pipeline demonstrating a real-time sentiment analysis system.

It uses Kafka for message streaming, MLflow for model versioning and registry, and a FastAPI service to perform live inference and store results in a PostgreSQL database.

## **🏛️ Architecture**

The data flows through the system as follows:

1. **producer**: A Python script that simulates a live data feed by sending random text messages (e.g., "I love this product\!") to a Kafka topic.  
2. **kafka**: The message broker that receives messages from the producer and holds them for the model\_service.  
3. **mlflow\_server**: Hosts the MLflow UI and Model Registry. This is where our trained sentiment models are stored, versioned, and aliased (e.g., as "champion").  
4. **model\_service**: A FastAPI application that:  
   * On startup, connects to the mlflow\_server and downloads the model marked with the "champion" alias.  
   * Consumes messages from the Kafka topic.  
   * Uses the loaded model to predict the sentiment of each message.  
   * Saves the message text, prediction, and timestamps to the postgres\_db.  
5. **postgres\_db**: The final destination database that stores all predictions for later analysis or to power a dashboard.

## **🛠️ Technology Stack**

* **Orchestration:** Docker & Docker Compose  
* **Streaming:** Kafka  
* **Model Registry:** MLflow (v2.9.2)  
* **Inference Service:** FastAPI (Python)  
* **Database:** PostgreSQL  
* **Producer:** Python  
* **Training:** Jupyter Notebook (nbconvert)

## **⚡ How to Run the Pipeline**

This pipeline has a two-phase setup:

1. **Phase 1: Start Infrastructure:** Run all services. The model\_service will intentionally fail and restart because no model exists yet.  
2. **Phase 2: Train & Deploy Model:** Run the training script to train a model and register it in MLflow, which "unlocks" the model\_service.

### **Prerequisites**

Before you begin, ensure your requirements.txt file (used by Dockerfile) includes the correct mlflow version to match the server and the nbconvert package for running the training script.

### **requirements.txt**

mlflow==2.9.2  
nbconvert

### **... other packages like fastapi, pandas, aiokafka, etc.**

### Phase 1: Start All Services

1\.  Build and start all services in detached mode:  
    \`\`\`bash  
    docker-compose up \-d \--build  
    \`\`\`

2\.  At this point, the \`model\_service\` will be in a crash loop. You can see this by running \`docker-compose logs \-f model\_service\`. It will show an error like \`Registered model alias champion not found\`. \*\*This is expected.\*\*

### Phase 2: Train and Deploy the First Model

You must now run the training script \*inside\* one of the running containers to register the first model.

1\.  Open a new terminal and "exec" into the \`producer\` container. We pass the \`MLFLOW\_TRACKING\_URI\` environment variable so the script knows where to find the MLflow server.  
    \`\`\`bash  
    docker-compose exec \-e MLFLOW\_TRACKING\_URI=<http://mlflow\_server:5000> producer /bin/bash  
    \`\`\`

2\.  Inside the container, convert the Jupyter Notebook to a Python script:  
    \`\`\`bash  
    jupyter nbconvert \--to script src/notebooks/01\_model\_training.ipynb  
    \`\`\`

3\.  Run the Python script to train and register the model:  
    \`\`\`bash  
    python src/notebooks/01\_model\_training.py  
    \`\`\`  
    You will see output as the script trains, logs metrics, and finally registers the model with the "champion" alias.

\---

## ✅ Verify It's Working

Once the training script is finished, the \`model\_service\` (on its next automatic restart) will successfully download the "champion" model and begin processing messages.

### 1\. Check the Service Logs

Check the logs for the \`model\_service\`:  
\`\`\`bash  
docker-compose logs \-f model\_service

You should now see new logs appearing every few seconds, like:

Model giga-flow-sentiment (Alias: champion) loaded successfully.  
Starting Kafka consumer...  
Kafka consumer connected successfully.  
...  
Received message: {'text': "I'm so frustrated with this app.", ...}  
Prediction: 'I'm so frustrated with this app.' \-\> Negative  
Prediction saved to database.

### **2\. Check the Database**

1. Connect to the PostgreSQL database:  
   docker-compose exec postgres\_db psql \-U gigaflow \-d sentiment\_db

2. Run a SQL query to see the predictions being saved:  
   SELECT text, sentiment\_label, processed\_at  
   FROM sentiment\_predictions  
   ORDER BY processed\_at DESC  
   LIMIT 10;

3. Type \\q to exit psql and exit to leave the container.

## **🔄 Deploying a New Model (The MLOps Loop)**

Now that the pipeline is running, deploying an improved model is simple:

1. Make your changes to the 01\_model\_training.ipynb notebook (e.g., add more data, use a better model).  
2. Repeat **Phase 2** (exec into the container, convert, and run the script). The script will register a new model version and update the "champion" alias to point to it.  
3. Restart the model\_service to force it to load the new model immediately:  
   docker-compose restart model\_service  
