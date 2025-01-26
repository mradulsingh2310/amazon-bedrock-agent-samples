# Property Assistant Agent

This is a multi-agent system that helps users with property-related queries using Amazon Bedrock. The system consists of specialized agents that work together to provide comprehensive property information.

## Architecture

The system consists of three main agents:
1. **Property Agent**: Handles queries about property details, features, and locations
2. **Payment Agent**: Handles queries about prices, payment terms, and financial aspects
3. **Supervisor Agent**: Orchestrates the collaboration between specialized agents

## Prerequisites

- Python 3.8+
- AWS Account with access to Amazon Bedrock
- Knowledge base set up in Amazon Bedrock

## Setup

1. Create a `.env` file in the project root with your AWS credentials:
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_SESSION_TOKEN=your_session_token
AWS_DEFAULT_REGION=your_region
KNOWLEDGE_BASE_ID=your_knowledge_base_id
KNOWLEDGE_BASE_DESCRIPTION=your_knowledge_base_description
MODEL_ID=your_model_id
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the assistant with a query:
```bash
python main.py --query "Tell me about 2-bedroom properties in downtown"
```

Additional options:
- `--recreate_agents`: Set to "false" to reuse existing agents (default: "true")
- `--trace_level`: Set trace level to "core", "outline", or "all" (default: "core")

## Example Queries

1. Pet Information:
```bash
python main.py --query "What are the pet policies?"
```

2. Payment Information:
```bash
python main.py --query "What are the payment terms for the property at 123 Main St?"
```

3. Combined Information:
```bash
python main.py --query "Tell me about pet policies and per month property rent charges"
``` 