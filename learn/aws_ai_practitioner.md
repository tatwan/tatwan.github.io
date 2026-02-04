The first table covers the AWS AI & Machine Learning Services (Domains 1, 2, and 3). The second table covers the Infrastructure, Security, and Data Services (Domains 4 and 5) that act as the backbone for AI workloads.
1. Essential AWS AI & ML Services
This list includes the "In-Scope" AI services found in the exam guide.
| Service Category | Service Name | Key Function for the Exam |
| :--- | :--- | :--- |
| **Generative AI** | **Amazon Bedrock** | A fully managed, serverless service to access Foundation Models (FMs) from Amazon (Titan) and startups (Anthropic, Cohere, etc.) via API. Used for building GenAI apps without managing infrastructure [1], [2]. |
| **Generative AI** | **Amazon Q** | A GenAI-powered assistant for businesses (Q Business) and developers (Q Developer) to generate code, answer questions from internal data, and debug [3], [4]. |
| **Generative AI** | **Amazon Nova** | A new series of foundation models developed by Amazon (referenced in the updated exam scope) [5], [6]. |
| **Machine Learning** | **Amazon SageMaker AI** | The comprehensive platform to build, train, and deploy custom ML models. Includes **JumpStart** (model hub), **Canvas** (no-code ML), **Clarify** (bias detection), and **Pipelines** [7], [8], [9]. |
| **Computer Vision** | **Amazon Rekognition** | Pre-trained vision service for image and video analysis, facial recognition, and content moderation [10], [11]. |
| **NLP & Text** | **Amazon Comprehend** | Natural Language Processing (NLP) service to extract insights, sentiment, and entities from text documents [12], [13]. |
| **NLP & Text** | **Amazon Textract** | Extract text, handwriting, and data tables from scanned documents (OCR+) [12], [14]. |
| **NLP & Text** | **Amazon Translate** | Fluent, neural machine translation service [12], [14]. |
| **Speech & Audio** | **Amazon Transcribe** | Automatic Speech Recognition (ASR) service that converts speech into text [12], [14]. |
| **Speech & Audio** | **Amazon Polly** | Text-to-Speech (TTS) service that turns text into lifelike speech [12], [14]. |
| **Chatbots** | **Amazon Lex** | Service for building conversational interfaces (chatbots) using voice and text [12], [14]. |
| **Search** | **Amazon Kendra** | Intelligent enterprise search service powered by ML to find content across data silos [10], [11]. |
| **Personalization** | **Amazon Personalize** | Real-time personalization and recommendation engine [10], [11]. |
| **Human Review** | **Amazon A2I (Augmented AI)** | Implements human review workflows for low-confidence ML predictions [15], [16]. |
| **Fraud** | **Amazon Fraud Detector** | Detects potentially fraudulent online activities (like fake accounts or payments) [17], [18]. |

----
2. Essential Support Services (Infrastructure, Security, Data)
You must understand how these services secure, store data for, and monitor AI workloads.
| Service Category | Service Name | Role in AI Workloads |
| :--- | :--- | :--- |
| **Compute** | **Amazon EC2** | Provides the underlying compute power. You must know instance types like **Trn1/Trn2** (Training) and **Inf1/Inf2** (Inference) utilized for deep learning [19], [20]. |
| **Compute** | **AWS Lambda** | Serverless compute used to orchestrate AI workflows (e.g., triggering a model when a file is uploaded) [21], [22]. |
| **Storage** | **Amazon S3** | The primary data lake storage for training data, model artifacts, and knowledge bases. Supports encryption and versioning [23], [24]. |
| **Database (Vector)**| **Amazon OpenSearch Service** | The recommended vector database for RAG (Retrieval Augmented Generation) to store embeddings for Amazon Bedrock [25], [26]. |
| **Database (Vector)**| **Amazon Aurora / RDS** | Relational databases that also support vector storage (via `pgvector`) for ML applications [25], [5]. |
| **Identity** | **AWS IAM** | Controls access to AI services. You must understand **Roles** (for services like SageMaker to access S3) and **Policies** (Least Privilege) [27], [28]. |
| **Security** | **Amazon Macie** | Uses ML to discover and protect sensitive data (PII) in S3 buckets to prevent it from being used in training improperly [29], [30]. |
| **Security** | **AWS KMS** | Key Management Service. Used to encrypt training data and model artifacts at rest [23], [24]. |
| **Networking** | **Amazon VPC & PrivateLink** | Secures network traffic. **PrivateLink** keeps traffic between your VPC and AWS AI services (like Bedrock) off the public internet [31], [32]. |
| **Compliance** | **AWS Audit Manager** | Automates evidence collection for audits. Includes frameworks specifically for Generative AI best practices [33], [34]. |
| **Compliance** | **AWS Artifact** | The portal to download AWS compliance reports (e.g., ISO, SOC) to prove AWS meets regulatory standards [35], [36]. |
| **Governance** | **AWS Config** | Tracks configuration changes of resources (e.g., did someone change a SageMaker endpoint configuration?) [37], [38]. |
| **Monitoring** | **Amazon CloudWatch** | Monitors operational metrics (e.g., number of invocations, latency, errors) for Bedrock and SageMaker [30], [39]. |
| **Logging** | **AWS CloudTrail** | Logs API calls (who did what). Essential for governance to see who accessed a model or changed a prompt [37], [38]. |

---

Essential Amazon SageMaker AI Components

| Component                         | Function & Exam Relevance                                    |
| --------------------------------- | ------------------------------------------------------------ |
| **SageMaker JumpStart**           | **Model Hub & Fine-tuning.**<br>Provides access to pre-trained Foundation Models (FMs) and built-in algorithms. Allows "1-click" deployment and fine-tuning of models without starting from scratch. Key for **Generative AI** tasks (Domain 3),,,. |
| **SageMaker Canvas**              | **No-Code/Low-Code ML.**<br>A visual interface for business analysts to build models (prediction, classification) and chat with LLMs without writing code. Often the answer for "democratizing AI" or "reducing technical barriers",,. |
| **SageMaker Autopilot**           | **Automated ML (AutoML).**<br>Automatically builds, trains, and tunes models based on your data. It handles algorithm selection, data preprocessing, and hyperparameter optimization transparently. Used for regression and classification tasks on tabular data,,. |
| **SageMaker Clarify**             | **Bias & Explainability.**<br>Detects bias in training data (pre-training) and in deployed models (post-training). Provides "explainability" (feature attribution) to help understand *why* a model made a specific prediction (Task 4.2),,. |
| **SageMaker Model Monitor**       | **Production Monitoring.**<br>Continuously monitors deployed models for **data drift**, **model quality** (accuracy) degradation, and bias drift. It alerts you when the statistical nature of live data deviates from training data,,. |
| **SageMaker Data Wrangler**       | **Data Preparation.**<br>A low-code tool to import, visualize, clean, and transform data before training. It accelerates data preparation using a visual interface,,. |
| **SageMaker Feature Store**       | **Feature Management.**<br>A centralized repository to store, discover, and share ML features (curated data attributes) so they can be reused across teams and models. Ensures consistency between training and inference (online/offline stores),,. |
| **SageMaker Ground Truth**        | **Data Labeling.**<br>Manages data labeling workflows using human workers (private, vendor, or Mechanical Turk) to create high-quality labeled datasets for supervised learning,. |
| **SageMaker Pipelines**           | **CI/CD & Orchestration.**<br>A workflow orchestration service to automate end-to-end ML steps (data prep -> train -> deploy). Critical for **MLOps** (Machine Learning Operations),. |
| **SageMaker Model Registry**      | **Model Governance.**<br>Catalog models, manage versions, and track metadata (like approval status: "Approved" or "Rejected") for deployment. Central to MLOps governance,. |
| **SageMaker Model Cards**         | **Documentation.**<br>Creates a "fact sheet" for your model. It documents critical details like intended use, risk ratings, training parameters, and performance observations for auditing and transparency (Task 5.1),,. |
| **SageMaker Role Manager**        | **Access Control.**<br>Helps administrators define custom permissions and IAM roles for ML activities (e.g., "Data Scientist" persona) to ensure least-privilege access,. |
| **SageMaker ML Lineage Tracking** | **Audit & Traceability.**<br>Tracks the history of a model, including which data was used, who trained it, and where it was deployed. Essential for compliance and reproducing workflows,,. |
| **SageMaker Inference**           | **Deployment Options.**<br>• **Real-time:** Low latency, constant traffic.<br>• **Serverless:** Intermittent traffic, pay-per-use.<br>• **Asynchronous:** Large payloads, long processing times.<br>• **Batch Transform:** Offline processing of large datasets,,. |
| **SageMaker Neo**                 | **Edge Optimization.**<br>Compiles and optimizes models to run efficiently on specific hardware and edge devices (IoT),. |

Summary of Differences for the Exam

• **Drift vs. Bias:** If the question asks about **drift** (performance getting worse over time), the answer is **Model Monitor**. If it asks about **fairness/bias** or **explainability**, the answer is **Clarify**,.

• **Code vs. No-Code:** If the user has no ML expertise, look for **Canvas** (No-code) or **Autopilot** (AutoML),.

• **Documentation:** If the question asks about documenting "intended uses" or "risk ratings" for compliance, the answer is **Model Cards**,.
