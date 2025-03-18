"""
API routes for the Financial QA system.
"""

from fastapi import APIRouter, HTTPException

from src.api.models import FinQARequest, FinQAResponse
from src.core.workflow import process_financial_question

# Create router
router = APIRouter(
    prefix="/financial-qa",
    tags=["financial-qa"],
)

@router.post("/questions", response_model=FinQAResponse)
def process_question(request: FinQARequest):
    """
    Process a financial question and return the answer.
    
    Args:
        request: The financial question and context data
        
    Returns:
        The answer and solution steps
    """
    try:
        # Prepare pre_context and post_context
        pre_context = ""
        post_context = ""
        
        if request.pre_text:
            pre_context = "\n".join(request.pre_text)
        
        if request.post_text:
            post_context = "\n".join(request.post_text)
        
        # Pass table data directly as list of lists
        table_data = request.table if request.table else None
        
        # Process the question
        result = process_financial_question(
            question=request.question,
            table_data=table_data,
            pre_context=pre_context,
            post_context=post_context
        )
        
        # Check for errors
        if "error" in result and result["error"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        # Handle any unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the question: {str(e)}") 