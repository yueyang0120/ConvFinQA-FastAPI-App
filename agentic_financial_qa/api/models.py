"""
Pydantic models for the Financial QA API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class FinQARequest(BaseModel):
    """
    Request model for the Financial QA API, based on the train.json format.
    """
    question: str = Field(..., description="The financial question to answer")
    pre_text: Optional[List[str]] = Field(None, description="Text before the table")
    post_text: Optional[List[str]] = Field(None, description="Text after the table")
    table: Optional[List[List[str]]] = Field(None, description="Table data as a list of rows")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "what was the percentage change in the net cash from operating activities from 2008 to 2009",
                "pre_text": [
                    "26 | 2009 annual report in fiscal 2008 , revenues in the credit union systems and services business segment increased 14% ( 14 % ) from fiscal 2007 .",
                    "all revenue components within the segment experienced growth during fiscal 2008 .",
                    "license revenue generated the largest dollar growth in revenue as episys ae , our flagship core processing system aimed at larger credit unions , experienced strong sales throughout the year .",
                    "support and service revenue , which is the largest component of total revenues for the credit union segment , experienced 34 percent growth in eft support and 10 percent growth in in-house support .",
                    "gross profit in this business segment increased $ 9344 in fiscal 2008 compared to fiscal 2007 , due primarily to the increase in license revenue , which carries the highest margins .",
                    "liquidity and capital resources we have historically generated positive cash flow from operations and have generally used funds generated from operations and short-term borrowings on our revolving credit facility to meet capital requirements .",
                    "we expect this trend to continue in the future .",
                    "the company 2019s cash and cash equivalents increased to $ 118251 at june 30 , 2009 from $ 65565 at june 30 , 2008 .",
                    "the following table summarizes net cash from operating activities in the statement of cash flows : 2009 2008 2007 ."
                ],
                "post_text": [
                    "year ended june 30 , cash provided by operations increased $ 25587 to $ 206588 for the fiscal year ended june 30 , 2009 as compared to $ 181001 for the fiscal year ended june 30 , 2008 .",
                    "this increase is primarily attributable to a decrease in receivables compared to the same period a year ago of $ 21214 ."
                ],
                "table": [
                    ["", "Year ended June 30, 2009"],
                    ["2008", "2007"],
                    ["Net income", "$103,102", "$104,222", "$104,681"],
                    ["Non-cash expenses", "74,397", "70,420", "56,348"],
                    ["Change in receivables", "21,214", "-2,913", "-28,853"],
                    ["Change in deferred revenue", "21,943", "5,100", "24,576"],
                    ["Change in other assets and liabilities", "-14,068", "4,172", "17,495"],
                    ["Net cash from operating activities", "$206,588", "$181,001", "$174,247"]
                ]
            }
        }
    }


class Step(BaseModel):
    """A step in the solution process."""
    op: str = Field(..., description="Operation type")
    arg1: str = Field(..., description="First argument")
    arg2: str = Field(..., description="Second argument")
    res: str = Field(..., description="Result of the operation")


class SolutionPlanVariable(BaseModel):
    """A variable in the solution plan."""
    name: str = Field(..., description="Variable name")
    source: str = Field(..., description="Source of the variable (table/text)")
    identifier: str = Field(..., description="Extraction identifier")


class CalculationStep(BaseModel):
    """A calculation step in the solution plan."""
    description: str = Field(..., description="Description of the calculation")
    expression: str = Field(..., description="Mathematical expression")
    result_var: str = Field(..., description="Result variable name")


class SolutionPlan(BaseModel):
    """The solution plan for answering the question."""
    variables: List[SolutionPlanVariable] = Field(..., description="Variables needed for the solution")
    calculation_steps: List[CalculationStep] = Field(..., description="Calculation steps")


class FinQAResponse(BaseModel):
    """
    Response model for the Financial QA API.
    """
    answer: str = Field(..., description="The final answer to the question")
    steps: List[Step] = Field(..., description="Steps taken to derive the answer")
    variables: Dict[str, Any] = Field(..., description="Variables and their values")
    solution_plan: Optional[SolutionPlan] = Field(None, description="The solution plan") 