import asyncio
from workflow import DiagnosisWorkflow
import pprint
import re
import json
import os
import sys
import time
import signal
import argparse  # Add argparse import

# Add a timeout handler decorator
def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timeout (exceeded 30 minutes)")

async def opslen_rca(instruction=None, dataset=None, record_idx=None, model=None, groundtruth_reason=None):
    """
    RCA (Root Cause Analysis) entry function for the operations diagnosis system
    
    Args:
        instruction: User-provided diagnosis instruction, if None then use default instruction
        dataset: Dataset name, used for generating output filename
        record_idx: Record index, used for generating output filename
        
    Returns:
        str: Root cause analysis result
    """
    # Initialize final_result variable to prevent undefined variable access in exception handling
    final_result = None
    
    try:
        # Set 30-minute timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1800)  
        
        # Direct output to terminal without capturing
        print("[DEBUG] Starting workflow creation...")
        try:
            async with await DiagnosisWorkflow.create() as workflow:
                print("[DEBUG] Workflow created successfully")
                # If no instruction is provided, use default instruction
                if instruction is None:
                    user_query = "On March 10, 2021, between 15:00 and 15:30, two system failures were encountered. The components responsible for these failures and the reasons behind them are not yet known. Please identify the root cause components and the root cause reasons."
                else:
                    user_query = instruction
                    
                queried_issue = ""
                reference_books = [""]
                
                print(f"[Execution Start] Processing instruction: {user_query}")
                print(f"[Dataset: {dataset}, Record Index: {record_idx}]")
                print(f"[Timeout limit set: 30 minutes]")
                
                try:
                    start_time = time.time()
                    diagnosis_result = await workflow.run_diagnosis(
                        user_query=user_query,
                        queried_issue=queried_issue,
                        reference_books=reference_books
                    )
                    end_time = time.time()
                    print(f"[Execution Complete] Time taken: {end_time - start_time:.2f} seconds")
                    
                    print("[Main] Diagnosis Result:")
                    pprint.pprint(diagnosis_result)
                    
                    final_result = diagnosis_result["root_cause"]
                    if "```json" in final_result:
                        final_result = re.search(r"```json\n(.*)\n```", final_result, re.S).group(1).strip()
                    
                    print("--------------------------------Final Result--------------------------------")
                    print(final_result)
                    print("--------------------------------Final Result--------------------------------")
                except Exception as inner_e:
                    print(f"[ERROR] Error during diagnosis execution: {str(inner_e)}")
                    print(f"[ERROR] Error type: {type(inner_e).__name__}")
                    import traceback
                    print(f"[ERROR] Full traceback:")
                    traceback.print_exc()
                    final_result = None
        except Exception as workflow_e:
            print(f"[ERROR] Error during workflow creation: {str(workflow_e)}")
            print(f"[ERROR] Error type: {type(workflow_e).__name__}")
            import traceback
            traceback.print_exc()
            final_result = None
        
        # Disable timeout alarm
        signal.alarm(0)
        
        
        # If final_result is None (possibly due to ignored exceptions during code execution), use static prediction
        if final_result is None:
            print("--------------------------------final_result is empty--------------------------------")
            final_result = get_static_prediction()
            print("--------------------------------final_result is empty--------------------------------")
            
        return final_result
            
    except TimeoutError as te:
        print(f"Diagnosis process timeout: {str(te)}")
        # Return a timeout prediction result
        static_result = get_static_prediction()
        print(f"Returning static prediction result: {static_result}")
        return static_result
    except Exception as e:
        # Ensure timeout alarm is disabled
        signal.alarm(0)
        
        print(f"[ERROR] Error during diagnosis process: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Full traceback:")
        traceback.print_exc()
        # Return a static prediction result
        static_result = get_static_prediction()
        print(f"[FALLBACK] Returning static prediction result: {static_result}")
        return static_result

def get_static_prediction():
    """Return a static prediction result for testing the evaluation system"""
    return json.dumps({
        "root_cause_component": "NA",
        "root_cause_reason": "NA",
        "root_cause_time": "NA"
    }, ensure_ascii=False)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='OpsLens RCA System')
    parser.add_argument('--query', type=str, help='The RCA query to process')
    args = parser.parse_args()

    print("--------------------------------Execution Start--------------------------------")
    
    # Use the query from command line if provided, otherwise use default
    query = args.query if args.query else "On March 10, 2021, between 15:00 and 15:30, two system failures were encountered. The components responsible for these failures and the reasons behind them are not yet known. Please identify the root cause components and the root cause reasons."
    
    final_result = asyncio.run(opslen_rca(instruction=query))
    print("--------------------------------Final Result--------------------------------")
    print(final_result)
    print("--------------------------------Final Result--------------------------------")
    
    
    