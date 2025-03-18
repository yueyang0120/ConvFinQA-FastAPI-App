# Financial QA FastAPI App

A FastAPI-based server that answers financial questions using an agentic workflow.

## Features

- Process financial questions about text and tabular data
- Step-by-step solution planning and execution
- Detailed answer generation with explanations
- Offline batch inference and evaluation capabilities against the ConvFinQA dataset

## Solution Architecture

This Financial QA system implements a structured agentic workflow using LangGraph to decompose complex financial problem-solving into discrete, traceable steps. The design prioritizes both accuracy and explainability to various financial contexts.

### Agentic Workflow Implementation

The system uses `LangGraph` to orchestrate a directed graph of specialized reasoning steps:

```
START → create_solution_plan → extract_data → perform_calculations → generate_answer → END
```

Each stage has error handling that routes to END on failure, providing clear error messages.

#### 1. Planning Stage (`create_solution_plan`) 

- **Function**: Breaks down the question into a structured solution plan
- **Process**: Uses LLM to determine required variables and calculation steps
- **Output**: JSON plan specifying variables and math operations needed

#### 2. Extraction Stage (`extract_data`)

- **Function**: Gets numeric values from text and tables
- **Process**: Sends variable specs and source data to LLM
- **Output**: Dictionary of extracted values

#### 3. Calculation Stage (`perform_calculations`) 

- **Function**: Runs calculations from solution plan
- **Process**: Evaluates math expressions in restricted Python env
- **Output**: Results from each calculation step


#### 4. Answer Generation Stage (`generate_answer`)

- **Function**: Creates formatted answer
- **Process**: Uses context to determine number formatting
- **Output**: Clean, readable response


### State Management

The workflow uses a `FinQAState` TypedDict to maintain the changing state of the question-answering process:

```python
class FinQAState(TypedDict):
    question: str               # The financial question
    pre_context: str            # Text before the table
    post_context: str           # Text after the table
    table: Optional[Any]        # Table data
    solution_plan: Optional[Dict]  # Structured solution plan
    variables: Dict[str, Any]   # Extracted variables and results
    steps: List[Dict[str, Any]] # Detailed steps for traceability
    answer: Optional[str]       # Final formatted answer
    error: Optional[str]        # Error information if any
```

This setup makes it easy to track everything and explain each step clearly.

### Technical Stack & Design Choices

#### **Why LangGraph?**

LangGraph makes the workflow way more maintainable:

1. **Clear Flow**: You can easily follow how data moves through the system
2. **Smart Error Handling**: Gracefully routes errors to the right place
3. **Easy Updates**: Change individual parts without breaking everything
4. **Keeps Track**: Manages all the state so you don't have to

#### **Debugging Made Easy**

Every operation gets logged with its inputs and outputs, like this: 

```json
// Example 1: Variable extraction operation
{
  "op": "extract",
  "arg1": "revenue_2020",
  "arg2": "table[3,2]",
  "res": "34576"
}

// Example 2: Calculation operation
{
  "op": "minus2-1",
  "arg1": "revenue_2021",
  "arg2": "revenue_2020",
  "res": "5400"
}
```
The logging structure makes debugging and logic tracing way easier:

1. Every operation is logged with full input/output details
2. Track exactly how each value was calculated
3. Run the same inputs again to verify fixes
4. The logs tell you exactly what happened and why

When something breaks, you get clear error messages pointing to the exact issue. No more guessing which part of the workflow failed - you'll know precisely where to look.

#### **Why LLMs Instead of Pandas When Doing Data Extraction with Tables?**

I chose LLM-based table parsing over Pandas because it handles real-world financial data better:

1. **Deals with Messy Tables**: Handles merged cells and weird formats that would break traditional parsers
2. **Gets the Context**: Understands how table data relates to surrounding text
3. **Makes Smart Guesses**: Can figure out unclear values using financial knowledge
4. **No Format Headaches**: Works with different table styles without complex preprocessing


## Project Structure

```
.
├── README.md                   # Project documentation
├── pyproject.toml              # Poetry project definition
├── poetry.lock                 # Poetry dependency lockfile
├── server.py                   # Server entry point
├── .env                        # Environment variables (API keys)
├── src/                        # Source code
│   ├── api/                    # API layer
│   │   ├── models.py           # Pydantic models
│   │   └── routes.py           # API routes
│   ├── core/                   # Core business logic
│   │   └── workflow.py         # Financial QA workflow
│   └── main.py                 # FastAPI app definition
├── scripts/                    # Utility scripts
│   └── batch_inference_eval_finqa.py  # Batch inference and evaluation script
├── tests/                      # Test files
│   └── test_api.py             # API tests
├── data/                       # Data directory
│   └── ConvFinQA/              # ConvFinQA dataset
```

## Installation and Project Setup

### Installing Poetry

This project uses Poetry for dependency management. If you don't have Poetry installed, you can install it using one of the following methods:

#### On macOS/Linux:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### On Windows:

```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

After installation, you may need to add Poetry to your PATH. The installer will provide instructions for doing this.

### Setting Up the Project

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2. Install dependencies with Poetry:

    ```bash
    poetry install
    ```

3. Create a `.env` file in the project root directory with your OpenAI API key:

    ```bash
    echo 'OPENAI_API_KEY="your-api-key-here"' > .env
    ```

    Replace `your-api-key-here` with your actual OpenAI API key. This key is required for the LLM to function.

4. Activate the virtual environment:

    ```bash
    poetry shell
    ```

## Running the API Server

Run the API server with default settings:

```bash
poetry run python server.py
```

Or specify custom host and port:

```bash
poetry run python server.py --host 127.0.0.1 --port 5000
```

## API Endpoints

### `POST /financial-qa/questions`

Process a financial question and return the answer.

#### Request Format

```json
{
  "question": "What was the percentage change in operating income between 2017 and 2018?",
  "pre_text": [
    "Financial Highlights ($ in millions except per share amounts)",
    "We delivered another year of strong financial performance in fiscal year 2018."
  ],
  "post_text": [
    "Operating income increased primarily due to growth across each of our business segments."
  ],
  "table": [
    ["Year", "2016", "2017", "2018"],
    ["Revenue", "91,154", "110,360", "125,843"],
    ["Operating Income", "26,147", "34,576", "39,240"]
  ]
}
```

#### Response Format

```json
{
  "answer": "13.5%",
  "steps": [
    {
      "op": "plan",
      "arg1": "What was the percentage change in operating income between 2017 and 2018?",
      "arg2": "",
      "res": "solution_plan_created"
    },
    {
      "op": "extract",
      "arg1": "income_2017",
      "arg2": "table",
      "res": "34576"
    },
    {
      "op": "extract",
      "arg1": "income_2018",
      "arg2": "table",
      "res": "39240"
    },
    {
      "op": "percentage_change",
      "arg1": "income_2018",
      "arg2": "income_2017",
      "res": "13.5"
    },
    {
      "op": "answer",
      "arg1": "Format the percentage change as a percentage",
      "arg2": "",
      "res": "13.5%"
    }
  ],
  "variables": {
    "income_2017": 34576,
    "income_2018": 39240,
    "percentage_change": 13.5
  },
  "solution_plan": {
    "variables": [
      {
        "name": "income_2017",
        "source": "table",
        "identifier": "row with 'Operating Income' and column '2017'"
      },
      {
        "name": "income_2018",
        "source": "table",
        "identifier": "row with 'Operating Income' and column '2018'"
      }
    ],
    "calculation_steps": [
      {
        "description": "Calculate percentage change in operating income",
        "expression": "(income_2018 - income_2017) / income_2017 * 100",
        "result_var": "percentage_change"
      }
    ]
  }
}
```

### `GET /health`

Health check endpoint to verify the API is running.

#### Response

```json
{
  "status": "healthy",
  "service": "Financial QA API"
}
```

## Example Usage with Python

```python
import requests
import json

# Define the API endpoint
api_url = "http://localhost:8000/financial-qa/questions"

# Create a request payload
payload = {
  "question": "What was the percentage change in operating income between 2017 and 2018?",
  "pre_text": [
    "Financial Highlights ($ in millions except per share amounts)",
    "We delivered another year of strong financial performance in fiscal year 2018."
  ],
  "post_text": [
    "Operating income increased primarily due to growth across each of our business segments."
  ],
  "table": [
    ["Year", "2016", "2017", "2018"],
    ["Revenue", "91,154", "110,360", "125,843"],
    ["Operating Income", "26,147", "34,576", "39,240"]
  ]
}

# Send the request
response = requests.post(api_url, json=payload)

# Print the response
if response.status_code == 200:
    result = response.json()
    print(f"Answer: {result['answer']}")
    print(f"Steps: {json.dumps(result['steps'], indent=2)}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

## Offline Batch Inference and Evaluation

The project includes a batch inference and evaluation script that tests the Financial QA system against the ConvFinQA dataset (train.json).

### Running the Batch Inference and Evaluation

```bash
# Run evaluation on the first 10 examples
poetry run python scripts/batch_inference_eval_finqa.py --num_examples 10

# Run evaluation on all examples and save results to a file
poetry run python scripts/batch_inference_eval_finqa.py --output_file results.json
```

### Evaluation Metrics

The evaluation script implements three complementary comparison strategies:

1. **Exact Match**: Checks if the normalized predicted answer exactly matches the normalized expected answer.

2. **Precision-Based Match**: Compares numbers based on their common precision level:
   
   - Identifies the precision level of both numbers (number of decimal places)
   - Uses the minimum precision between the two for comparison
   - Rounds both values to this common precision before comparing
   - Prevents false mismatches caused by insignificant trailing digits

3. **Scale-Aware Tolerance Match**: Applies different tolerance thresholds based on the value's magnitude:
   
   - Small values (< 1.0): Uses fixed tolerance of ±0.05
   - Large values (≥ 100.0): Uses percentage-based tolerance of ±1%
   - Medium values: Uses a balanced approach that scales with the value (weighted sum)
   - Ensures appropriate comparison across the full range of financial values

These complementary metrics together provide a comprehensive assessment of answer quality, recognizing that financial calculations may differ in presentation without being substantively wrong.

### Evaluation Results & Analysis

#### Current Performance Metrics

Based on running the evaluation on the first 100 examples from the ConvFinQA dataset `train.json`:

```
===== EVALUATION SUMMARY =====
Total examples: 100
Exact matches: 21 (21.00%)
Precision-based matches: 35 (35.00%)
Tolerance-based matches: 45 (45.00%)
Errors: 0 (0.00%)
===========================
```

#### Why These Three Metrics Were Chosen?

1. **Exact Match (21%)**
   - **Purpose**: Verifies character-for-character correctness after normalization
   - **Use Case**: Best for categorical answers or when precise formatting matters
   - **Technical Value**: Provides a baseline for strictest evaluation
   - **Limitation**: Too strict for numeric answers with formatting variations

2. **Precision Match (35%)**
   - **Purpose**: Compares values based on significant digits rather than string representation
   - **Implementation**: Rounds both numbers to the lower precision before comparison
   - **Use Case**: Handles different decimal place representations (e.g., "14.1%" vs "14.10%")
   - **Technical Value**: Eliminates false negatives from insignificant trailing digits

3. **Tolerance Match (45%)**
   - **Purpose**: Applies scale-appropriate tolerances for different numeric magnitudes
   - **Implementation**: Uses adaptive thresholds based on value size
   - **Use Case**: Accounts for legitimate rounding differences in large calculations
   - **Technical Value**: Most closely matches human judgment of correctness

#### Key Findings

1. **Error Handling is Robust**: 0% error rate indicates the system successfully processes all examples without crashing.

2. **Progressive Improvement Across Metrics**: The step-up from exact match (21%) to precision match (35%) to tolerance match (45%) suggests our system produces results that are numerically close but may differ in formatting or minor calculation details.

#### Current Limitations & Improvement Areas

1. **Formatting Consistency**: The gap between exact match and precision match (14%) indicates formatting inconsistencies that could be standardized.

2. **Accuracy Ceiling**: Even with the most forgiving metric, it achieves 45% accuracy, indicating substantial room for improvement in:
   - Better table data extraction
   - More robust handling of complex financial terminology
   - Enhanced reasoning for multi-step calculations

3. **Null Handling**: Some examples produce null answers rather than incorrect ones, indicating confidence thresholds may be too strict.

### Customizing Evaluation

The evaluation script supports several command-line arguments:

- `--dataset_path`: Path to the dataset file (default: data/ConvFinQA/data/train.json)
- `--num_examples`: Number of examples to evaluate (default: all examples)
- `--output_file`: File to save detailed evaluation results (default: None)

## Future Enhancements (for production usage)

To make this app production-ready, additional steps are required: 

- Authentication and authorization for API access
  - OAuth2 authentication flow
  - Role-based access control (RBAC)
  - API key management

- Batch processing capabilities
  - Parallel processing of multiple questions
  - Asynchronous API endpoints
  - Bulk import/export functionality

- Model enhancements
  - Support for additional LLM models
  - Configurable model parameters

- Performance optimizations
  - Query optimization
  - Load balancing
  - Rate limiting

- Enhanced evaluation framework
  - Additional evaluation metrics
  - Performance monitoring

- Infrastructure improvements
  - Kubernetes deployment
  - Horizontal pod autoscaling
  - CI/CD pipeline
  - Monitoring and logging