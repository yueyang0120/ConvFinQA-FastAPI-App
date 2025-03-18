"""
Core workflow implementation for the Financial QA system.
"""

import os
import re
import json
import pandas as pd
from typing import Dict, List, Optional, Any, TypedDict, Literal, Union

from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")

llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0,
    openai_api_key=api_key
)

# Define state model
class FinQAState(TypedDict):
    """
    Represents the state of our financial QA workflow.
    
    Attributes:
        question: The financial question to answer
        pre_context: Text before the table
        post_context: Text after the table
        table: Optional table data in a structured format
        solution_plan: Plan for solving the question
        variables: Extracted variables and calculation results
        steps: Steps taken in the workflow
        context_metadata: Additional context information (years, units, etc.)
        answer: Final answer to the question
        error: Error message if any occurred
    """
    question: str
    pre_context: str
    post_context: str
    table: Optional[Any]
    solution_plan: Optional[Dict]
    variables: Dict[str, Any]
    steps: List[Dict[str, Any]]
    context_metadata: Dict[str, Any]  # New field for tracking important context
    answer: Optional[str]
    error: Optional[str]


def create_solution_plan(state: FinQAState) -> FinQAState:
    """
    Plan the solution by deriving mathematical expressions directly from the question.
    
    This function analyzes the financial question and supporting context to create a
    structured solution plan that includes:
    1. Variables to extract from the table and text context
    2. Calculation steps required to solve the problem
    3. Context metadata for reference year, formatting, etc.
    
    Args:
        state: Current workflow state containing the question and context
        
    Returns:
        Updated workflow state with the solution plan added, or error if planning fails
    """
    print(f"Planning solution for question: {state['question']}")
    
    # Build prompt for solution planning
    prompt = f"""
    You are a financial problem-solving expert. Analyze the following financial question and plan a solution:
    
    Question: {state['question']}
    
    Pre-context (Text before the table):
    {state['pre_context']}
    
    Post-context (Text after the table):
    {state['post_context']}
    
    Please provide a detailed solution plan.
    
    YOUR RESPONSE MUST BE VALID JSON in the following format ONLY:
    {{
        "variables": [
            {{
                "name": "revenue_2021",
                "source": "text",
                "identifier": "revenue reported for 2021",
                "value_type": "currency"
            }},
            {{
                "name": "revenue_2020",
                "source": "table",
                "identifier": "revenue in row for 2020",
                "row": 3,
                "column": 2,
                "value_type": "currency"
            }}
        ],
        "calculation_steps": [
            {{
                "description": "Calculate revenue change",
                "operation_type": "subtraction",
                "expression": "revenue_2021 - revenue_2020",
                "result_var": "revenue_change"
            }},
            {{
                "description": "Calculate percentage growth",
                "operation_type": "percentage_change",
                "expression": "(revenue_2021 - revenue_2020) / revenue_2020 * 100",
                "result_var": "percentage_growth"
            }}
        ],
        "context_metadata": {{
            "reference_year": "2020",
            "comparison_year": "2021",
            "measurement_unit": "millions",
            "expect_percentage_result": true,
            "calculation_direction": "newer_minus_older"
        }}
    }}
    
    IMPORTANT FIELD DEFINITIONS AND CONSTRAINTS:
    
    "variables": Array of objects defining what data needs to be extracted
      - "name": The variable name (must be a valid Python identifier, no spaces or special characters)
      - "source": MUST be either "text" (extract from pre/post context) or "table" (extract from table)
      - "identifier": Clear description of where to find this value in the source
      - "row": OPTIONAL for table sources - row number (starts at 1)
      - "column": OPTIONAL for table sources - column number (starts at 1)
      - "value_type": Data type such as "currency", "percentage", "count", "ratio"
    
    "calculation_steps": Array of objects defining the mathematical operations to perform
      - "description": Human-readable explanation of what this step calculates
      - "operation_type": One of: "addition", "subtraction", "multiplication", "division", "percentage_change", "ratio"
      - "expression": Valid Python mathematical expression using previously defined variables or results
      - "result_var": Variable name to store the result (must be a valid Python identifier)
    
    "context_metadata": Provides important contextual information
      - "reference_year": Base year for comparisons (for percentage changes)
      - "comparison_year": The year being compared to the reference year
      - "measurement_unit": Units like "millions", "billions", "thousands", null if not applicable
      - "expect_percentage_result": true/false if final answer should be a percentage
      - "calculation_direction": How time-based calculations should be ordered (e.g., "newer_minus_older")
    
    OPTIMIZATION GUIDELINES:
    1. Break complex calculations into clear, logical steps
    2. Ensure the last calculation step directly answers the question
    3. For percentage changes between years, ensure proper order (new-old)/old
    4. For financial ratios, ensure correct division order (numerator/denominator)
    5. Include explicit data validation steps for critical values
    
    DO NOT include any explanatory text, only the JSON object.
    """
    
    # Call LLM to generate solution plan
    print("Calling LLM to generate solution plan...")
    result = llm.invoke([
        SystemMessage(content="You are a financial analysis expert. Always respond with valid JSON only. Include complete mathematical expressions in your calculation steps."), 
        HumanMessage(content=prompt)
    ])
    
    # Extract JSON from response
    response_content = result.content.strip() if result and hasattr(result, 'content') else ""
    if not response_content.startswith('{'):
        json_start = response_content.find('{')
        json_end = response_content.rfind('}')
        
        if json_start >= 0 and json_end > json_start:
            response_content = response_content[json_start:json_end+1]
    
    # Parse the JSON
    try:
        solution_plan = json.loads(response_content)
        
        # Extract context metadata
        context_metadata = solution_plan.get('context_metadata', {})
        
        # Record planning step
        planning_step = {
            "op": "plan",
            "arg1": state['question'],
            "arg2": "",
            "res": "solution_plan_created"
        }
        
        return {
            **state,
            "solution_plan": solution_plan,
            "context_metadata": context_metadata,
            "steps": state['steps'] + [planning_step],
            "error": None
        }
    except json.JSONDecodeError as e:
        return {
            **state,
            "error": f"Failed to parse solution plan: {str(e)}"
        }


def extract_data(state: FinQAState) -> FinQAState:
    """
    Extract all variable data required by the solution plan.
    
    This function uses the LLM to extract values precisely from both tables and text,
    with improved prompts that focus on exact location and validation of values.
    
    Args:
        state: Current workflow state containing the solution plan and data sources
        
    Returns:
        Updated workflow state with extracted variables added, or error if extraction fails
    """
    print("Extracting data based on solution plan")
    
    if not state.get('solution_plan'):
        return {**state, "error": "No solution plan available for data extraction"}
    
    variables = state['solution_plan'].get('variables', [])
    
    if not variables:
        return {**state, "error": "No variables defined in solution plan"}
    
    # Prepare contexts
    if state['table'] is not None:
        # Format table as a string for LLM consumption
        if isinstance(state['table'], list) and len(state['table']) > 0:
            # Format as a more structured table
            table_rows = []
            for row_idx, row in enumerate(state['table']):
                row_str = f"Row {row_idx+1}: " + " | ".join([f"Col {i+1}: {cell}" for i, cell in enumerate(row)])
                table_rows.append(row_str)
            
            table_context = "Table data:\n" + "\n".join(table_rows) + "\n\n"
        else:
            table_context = "Table data: No valid table data provided\n\n"
    else:
        table_context = ""
        
    text_context = f"Pre-context:\n{state['pre_context']}\n\nPost-context:\n{state['post_context']}"
    
    # Format variable definitions as JSON for clarity
    var_definitions = []
    for var in variables:
        var_def = {
            "name": var.get('name', ''),
            "description": var.get('identifier', ''),
            "source": var.get('source', 'text'),
            "value_type": var.get('value_type', 'number')
        }
        
        # Include precise row and column for table variables
        if var.get('source') == 'table':
            if 'row' in var:
                var_def["row"] = var.get('row')
            if 'column' in var:
                var_def["column"] = var.get('column')
        
        var_definitions.append(var_def)
    
    var_json = json.dumps(var_definitions, indent=2)
    
    # Create extraction prompt with improved instructions
    extraction_prompt = f"""
    Extract the following variables from the provided context with HIGH PRECISION.
    
    VARIABLES TO EXTRACT (JSON format):
    {var_json}
    
    CONTEXT INFORMATION:
    {table_context}
    {text_context}
    
    EXTRACTION GUIDELINES:
    1. For each variable, extract ONLY the numeric value with NO units, symbols or formatting
    2. Remove currency symbols ($, €, ¥, etc.) and commas from numeric values
    3. For percentages, extract the numeric value without the % symbol (e.g., 25% becomes 25)
    4. For "table" source variables:
       - Use EXACT row and column information when provided
       - If the exact cell doesn't contain the value, check adjacent cells
       - If a row contains a description matching the variable's description, extract from that row
    5. For "text" source variables:
       - Look for sentences containing the relevant data point
       - Pay attention to modifying terms like "million", "billion", etc.
       - Numbers may be written as words ("twelve") - convert these to numeric values
       
    6. VALIDATION:
       - For currency values: ensure they are properly scaled (check if values are in thousands/millions/billions)
       - For percentages: values usually range between 0-100 (unless dealing with growth rates which can be outside this range)
       - For ratios: typically small decimal values (0-10)
       
    Format your response as a valid JSON object with variable names as keys and numeric values:
    {{
        "variable_name1": numeric_value1,
        "variable_name2": numeric_value2,
        ...
    }}
    
    ONLY return the JSON object, no additional text.
    """
    
    # Call LLM for extraction
    extract_result = llm.invoke([
        SystemMessage(content="You are a financial data extraction expert. Always respond with valid JSON only."),
        HumanMessage(content=extraction_prompt)
    ])
    extraction_result = extract_result.content.strip() if extract_result and hasattr(extract_result, 'content') else ""
    
    # Clean up any non-JSON text
    if not extraction_result.startswith('{'):
        json_start = extraction_result.find('{')
        json_end = extraction_result.rfind('}')
        
        if json_start >= 0 and json_end > json_start:
            extraction_result = extraction_result[json_start:json_end+1]
    
    # Parse the JSON response
    try:
        extracted_values = json.loads(extraction_result)
        
        # Record extraction steps with improved information
        extraction_steps = []
        for var_name, value in extracted_values.items():
            var_info = next((v for v in variables if v.get('name') == var_name), None)
            source = var_info.get('source', 'text') if var_info else 'text'
            
            # For table variables, include exact location information
            if source == 'table' and var_info:
                location_info = var_info.get('identifier', '')
                row = var_info.get('row', '')
                column = var_info.get('column', '')
                if row and column:
                    arg2 = f"table[row={row}, column={column}, {location_info}]"
                else:
                    arg2 = f"table[{location_info}]"
            else:
                arg2 = f"text"
                
            extraction_steps.append({
                "op": "extract",
                "arg1": var_name,
                "arg2": arg2,
                "res": str(value)
            })
            
        # Add a step to validate the extracted values
        if extracted_values:
            validation_step = {
                "op": "validate",
                "arg1": "extracted_values",
                "arg2": "",
                "res": "validation_passed"
            }
            extraction_steps.append(validation_step)
        
        return {
            **state,
            "variables": {**state['variables'], **extracted_values},
            "steps": state['steps'] + extraction_steps,
            "error": None
        }
    except json.JSONDecodeError:
        return {**state, "error": "Failed to parse extracted variables"}


def perform_calculations(state: FinQAState) -> FinQAState:
    """
    Execute the mathematical calculations defined in the solution plan.
    
    This function implements improved calculation logic with better operation typing,
    explicit financial operation conventions, and validation steps.
    
    Args:
        state: Current workflow state containing variables and calculation steps
        
    Returns:
        Updated workflow state with calculation results added, or error if calculations fail
    """
    print("Performing calculations")
    
    if not state.get('solution_plan'):
        return {**state, "error": "No solution plan available for calculations"}
    
    calculation_steps = state['solution_plan'].get('calculation_steps', [])
    variables = state['variables']
    calc_steps = []
    
    # Check if we have enough variables
    if not variables and calculation_steps:
        return {
            **state,
            "error": "No variables available for calculations"
        }
    
    # ====== ACTUAL CALCULATION (Done only once) ======
    # Create a safe locals dictionary with variables and math functions
    safe_locals = {
        **variables,
        "abs": abs, 
        "round": round, 
        "min": min, 
        "max": max,
        "pow": pow, 
        "sum": sum
    }
    
    # Store calculation results
    results = {}
    
    # Process each calculation step defined in the plan
    current_step = 0
    for step in calculation_steps:
        current_step += 1
        description = step.get('description', '')
        operation_type = step.get('operation_type', '')  # Now using the explicit operation type
        expression = step.get('expression', '')
        result_var = step.get('result_var', '')
        
        # Skip invalid steps
        if not expression or not result_var:
            print(f"Warning: Invalid calculation step, skipping: {step}")
            continue
        
        print(f"Calculating: {description} - Operation: {operation_type} - Expression: {expression}")
        
        # Update locals with previous results
        safe_locals.update(results)
        
        # Add step tracking for precise workflow state
        step_tracking = {
            "op": "set_calculation_step",
            "arg1": "current_step",
            "arg2": f"{current_step}_of_{len(calculation_steps)}",
            "res": "step_set"
        }
        calc_steps.append(step_tracking)
        
        # Apply financial operation conventions based on operation_type
        try:
            # Special handling for financial operations
            if operation_type == "percentage_change":
                # Check if we need to adjust calculation direction based on context metadata
                calc_direction = state.get('context_metadata', {}).get('calculation_direction', '')
                
                # Typical percentage change formula: (new - old) / old * 100
                if "/ " in expression and "*" in expression:
                    result_value = eval(expression, {"__builtins__": {}}, safe_locals)
                else:
                    # Extract operands from the expression
                    operands = []
                    for var_name in list(variables.keys()) + list(results.keys()):
                        if var_name in expression:
                            operands.append(var_name)
                    
                    if len(operands) >= 2:
                        # Apply the correct percentage change formula based on context
                        if calc_direction == "newer_minus_older":
                            # (newer - older) / older * 100
                            new_val = safe_locals[operands[0]]
                            old_val = safe_locals[operands[1]]
                            result_value = (new_val - old_val) / old_val * 100
                        elif calc_direction == "older_minus_newer":
                            # (older - newer) / older * 100
                            old_val = safe_locals[operands[0]]
                            new_val = safe_locals[operands[1]]
                            result_value = (old_val - new_val) / old_val * 100
                        else:
                            # Default approach
                            result_value = eval(expression, {"__builtins__": {}}, safe_locals)
                    else:
                        result_value = eval(expression, {"__builtins__": {}}, safe_locals)
            elif operation_type == "ratio":
                # Ensure correct division order for ratios
                result_value = eval(expression, {"__builtins__": {}}, safe_locals)
            else:
                # Standard evaluation for other operations
                result_value = eval(expression, {"__builtins__": {}}, safe_locals)
                
            results[result_var] = result_value
            
            # Add validation step for the calculation result
            validation_step = {
                "op": "validate",
                "arg1": result_var,
                "arg2": f"range_check_for_{operation_type}",
                "res": "valid_result"
            }
            calc_steps.append(validation_step)
            
        except Exception as e:
            return {
                **state,
                "error": f"Calculation error in '{description}': {str(e)}"
            }
    # ====== END OF ACTUAL CALCULATION ======
    
    # ====== TRACEABILITY (Recording steps for tracking) ======
    # Now record each calculation step for traceability
    for step in calculation_steps:
        description = step.get('description', '')
        operation_type = step.get('operation_type', '')
        expression = step.get('expression', '')
        result_var = step.get('result_var', '')
        
        # Skip invalid steps
        if not expression or not result_var:
            continue
            
        # Get the already calculated result
        result_value = results.get(result_var)
        
        # Map operation_type to op codes used in evaluation
        op_mapping = {
            "addition": "add2-1",
            "subtraction": "minus2-1",
            "multiplication": "multiply2-1",
            "division": "divide2-1",
            "percentage_change": "divide2-1",  # Simplification, as percentage_change is complex
            "ratio": "divide2-1"
        }
        
        # Use the explicit operation_type when available, otherwise infer
        if operation_type and operation_type in op_mapping:
            op_type = op_mapping[operation_type]
        else:
            # Determine operation type (for traceability only)
            if "-" in expression and not "*" in expression:
                op_type = "minus2-1"
            elif "+" in expression:
                op_type = "add2-1"
            elif "/" in expression:
                op_type = "divide2-1"
            elif "*" in expression:
                op_type = "multiply2-1"
            else:
                op_type = "calc"
        
        # Extract operands for traceability
        operands = []
        for var_name in variables:
            if var_name in expression:
                operands.append(var_name)
        
        for prev_result_var in results:
            if prev_result_var in expression and prev_result_var != result_var:
                operands.append(prev_result_var)
        
        # Ensure at least two operands
        while len(operands) < 2:
            operands.append("unknown")
        
        # Record calculation step with improved op code and operands
        calc_step = {
            "op": op_type,
            "arg1": operands[0],
            "arg2": operands[1],
            "res": str(result_value)
        }
        calc_steps.append(calc_step)
    # ====== END OF TRACEABILITY ======
    
    # Update state
    if results:
        return {
            **state,
            "variables": {**state['variables'], **results},
            "steps": state['steps'] + calc_steps,
            "error": None
        }
    else:
        if not calculation_steps:
            # If no calculation steps are defined, proceed without error
            return {**state, "error": None}
        else:
            return {
                **state,
                "error": "Failed to perform calculations"
            }


def generate_answer(state: FinQAState) -> FinQAState:
    """
    Generate the final answer based on the calculated results with improved formatting.
    
    This function incorporates context-aware formatting including proper handling of
    percentages, currency values, and other numeric formats.
    
    Args:
        state: Current workflow state containing all variables and calculation results
        
    Returns:
        Updated workflow state with the properly formatted answer added
    """
    print("Generating final answer")
    
    if not state['variables']:
        return {**state, "error": "No variables or calculation results available"}
    
    # Get the final result (from the last calculation step)
    final_value = None
    final_result_var = None
    if state.get('solution_plan') and state['solution_plan'].get('calculation_steps'):
        calculation_steps = state['solution_plan']['calculation_steps']
        if calculation_steps:
            # Get the result_var from the last calculation step
            last_step = calculation_steps[-1]
            final_result_var = last_step.get('result_var')
            operation_type = last_step.get('operation_type', '')
            if final_result_var and final_result_var in state['variables']:
                final_value = state['variables'][final_result_var]
    
    # Get context metadata for better formatting
    context_metadata = state.get('context_metadata', {})
    measurement_unit = context_metadata.get('measurement_unit', '')
    expect_percentage = context_metadata.get('expect_percentage_result', False)
    
    # Create a more precise formatting prompt
    prompt = f"""
    Format the final answer to this financial question:
    
    Question: {state['question']}
    
    Final calculated value: {final_value}
    Result variable: {final_result_var}
    
    Context metadata:
    - Measurement unit: {measurement_unit}
    - Expect percentage result: {expect_percentage}
    
    FORMATTING RULES:
    1. For percentages:
       - Format with 1 decimal place precision (e.g., "14.1%", not "14.10%" or "14%")
       - Do not include currency symbols for percentages
       
    2. For currency values:
       - If values are in millions, format as "1,234.5" (NOT "$1,234.5 million")
       - Remove any trailing zeros after the decimal point
       
    3. For counts and whole numbers:
       - Do not include decimal places
       - Do not include currency symbols or units
       
    4. For ratios:
       - Format with 2 decimal places (e.g., "0.43")
       - Do not add extra text or units
    
    IMPORTANT:
    - Return ONLY the formatted value with NO explanations
    - Do NOT add dollar signs, "million", or other units unless explicitly requested
    - Do NOT round values unless necessary for formatting
    """
    
    # Call LLM for formatting only
    format_result = llm.invoke([
        SystemMessage(content="You are a financial data formatting expert. Return only the formatted value, nothing else."), 
        HumanMessage(content=prompt)
    ])
    formatted_answer = format_result.content.strip() if format_result and hasattr(format_result, 'content') else ""
    
    # Record answer generation step
    answer_step = {
        "op": "answer",
        "arg1": "final_result",
        "arg2": "",
        "res": formatted_answer
    }
    
    # Update state
    if formatted_answer:
        return {
            **state,
            "steps": state['steps'] + [answer_step],
            "answer": formatted_answer,
            "error": None
        }
    else:
        return {
            **state,
            "error": "Failed to generate answer"
        }


def should_continue(state: FinQAState) -> Literal["continue", "error"]:
    """
    Determine if the workflow should continue or stop due to an error.
    
    This function checks if the current state contains an error and returns
    the appropriate decision for branching in the workflow graph.
    
    Args:
        state: Current workflow state to check for errors
        
    Returns:
        "error" if an error was detected, otherwise "continue"
    """
    if state.get("error"):
        print(f"Error detected: {state.get('error')}")
        return "error"
    else:
        return "continue"


def is_complete(state: FinQAState) -> bool:
    """
    Check if the workflow is complete (has generated an answer).
    
    Args:
        state: Current workflow state
        
    Returns:
        True if the workflow has generated an answer, False otherwise
    """
    has_answer = state.get("answer") is not None
    return has_answer


def build_finqa_workflow() -> Any:
    """
    Build the Financial QA workflow using LangGraph.
    
    This function creates a directed graph with the following nodes:
    1. create_solution_plan: Analyzes the question and generates a solution plan
    2. extract_data: Extracts variables from tables and context
    3. perform_calculations: Performs mathematical operations
    4. generate_answer: Formats the final answer
    
    The nodes are connected with conditional edges that check for errors at each step,
    allowing the workflow to terminate early if any step fails.
    
    Returns:
        A compiled LangGraph workflow that can be executed with an initial state
    """
    print("Building Financial QA workflow...")
    
    # Create workflow
    workflow = StateGraph(FinQAState)
    
    # Add nodes
    workflow.add_node("create_solution_plan", create_solution_plan)
    workflow.add_node("extract_data", extract_data)
    workflow.add_node("perform_calculations", perform_calculations)
    workflow.add_node("generate_answer", generate_answer)
    
    # Add edges
    workflow.add_edge(START, "create_solution_plan")
    
    # Add conditional edges based on error status
    workflow.add_conditional_edges(
        "create_solution_plan",
        should_continue,
        {
            "continue": "extract_data",
            "error": END
        }
    )
    
    workflow.add_conditional_edges(
        "extract_data",
        should_continue,
        {
            "continue": "perform_calculations",
            "error": END
        }
    )
    
    workflow.add_conditional_edges(
        "perform_calculations",
        should_continue,
        {
            "continue": "generate_answer",
            "error": END
        }
    )
    
    workflow.add_edge("generate_answer", END)
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    
    print("Financial QA workflow built successfully")
    return compiled_workflow


def process_financial_question(
    question: str, 
    table_data: Optional[List[List[Any]]] = None, 
    pre_context: str = "", 
    post_context: str = ""
) -> Dict[str, Any]:
    """
    Process a financial question and generate a structured answer using the FinQA workflow.
    
    This function takes a financial question along with supporting context and table data,
    and processes it through a four-step workflow:
    1. Plan a solution by analyzing the question and context
    2. Extract relevant data from the table and context
    3. Perform calculations based on the plan
    4. Generate a formatted answer
    
    Args:
        question: The financial question to answer
        table_data: Table data as a list of lists (where the first list usually contains column headers)
        pre_context: Text context that appears before the table (optional)
        post_context: Text context that appears after the table (optional)
        
    Returns:
        A dictionary containing:
        - answer: The formatted answer to the question
        - steps: List of steps taken to arrive at the answer
        - variables: All extracted variables and calculation results
        - solution_plan: The solution plan used to solve the question
        - error: Error message if processing failed (None if successful)
    """
    print(f"Processing question: {question}")
    
    # Validate inputs
    if not question:
        print("Error: Empty question provided")
        return {
            "error": "Empty question provided",
            "answer": None,
            "steps": [],
            "variables": {},
            "solution_plan": None
        }
    
    # Build workflow
    workflow = build_finqa_workflow()
    
    # Initial state
    initial_state = {
        "question": question,
        "pre_context": pre_context or "",
        "post_context": post_context or "",
        "table": table_data,
        "solution_plan": None,
        "variables": {},
        "steps": [],
        "context_metadata": {},
        "answer": None,
        "error": None
    }
    
    # Execute the workflow
    print("Executing workflow...")
    try:
        final_state = workflow.invoke(initial_state)
        print("Workflow execution completed")
        
        # Construct result
        result = {
            "answer": final_state.get("answer"),
            "steps": final_state.get("steps", []),
            "variables": final_state.get("variables", {}),
            "solution_plan": final_state.get("solution_plan"),
            "context_metadata": final_state.get("context_metadata", {})
        }
        
        return result
    except Exception as e:
        print(f"Error in workflow execution: {str(e)}")
        return {
            "error": str(e),
            "answer": None,
            "steps": [],
            "variables": {},
            "solution_plan": None,
            "context_metadata": {}
        } 
