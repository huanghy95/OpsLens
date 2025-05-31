import sys
import os
import re
import argparse
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)




def evaluate(prediction:str, scoring_points:str):
    """
    Evaluate single JSON-like prediction with corresponding scoring points
        args:
            prediction: str, the prediction (JSON-like string)
            scoring_points: str, the scoring points string
        returns:
            passing_criteria: list, matched criteria
            failing_criteria: list, unmatched criteria
            binary_score: float, original binary score
            is_top1_match: bool, whether the prediction matches top1 criteria completely
            is_top3_match: bool, whether the prediction matches top3 criteria completely
            is_top1_partial: bool, whether the prediction partially matches top1 criteria
            is_top3_partial: bool, whether the prediction partially matches top3 criteria
    """

    import itertools

    # Try to parse JSON format
    try:
        pred_json = json.loads(prediction)
        predict_results = []
        # Handle possible nested structure
        for key, value in pred_json.items():
            if isinstance(value, dict):
                predict_results.append(value)
    except Exception as e:
        # Fall back to regex matching
        predict_pattern = (
            r'{\s*'
            r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
            r'(?:"root cause component":\s*"(.*?)")?,?\s*'
            r'(?:"root cause reason":\s*"(.*?)")?\s*}'
        )
        predict_matches = re.findall(predict_pattern, prediction)
        predict_results = []
        try:
            for match in predict_matches:
                if len(match) >= 3:  # Ensure match result has at least 3 elements
                    datetime_str, component, reason = match
                    predict_results.append({
                        "root cause occurrence datetime": datetime_str,
                        "root cause component": component,
                        "root cause reason": reason
                    })
                else:
                    # If match result is incomplete, create dictionary with available fields
                    result_dict = {}
                    if len(match) > 0:
                        result_dict["root cause occurrence datetime"] = match[0]
                    if len(match) > 1:
                        result_dict["root cause component"] = match[1]
                    if len(match) > 2:
                        result_dict["root cause reason"] = match[2]
                    predict_results.append(result_dict)
        except Exception as e:
            print(f"Regex matching processing error: {e}")
            # If regex matching also fails, create an empty result
            predict_results = [{
                "root cause occurrence datetime": "",
                "root cause component": "",
                "root cause reason": ""
            }]

    prediction_length = len(predict_results)

    component_pattern = r"The (?:\d+-th|only) predicted root cause component is ([^.\n(]+)"
    reason_pattern = r"The (?:\d+-th|only) predicted root cause reason is ([^.\n(]+)"
    time_pattern = r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i.e., <=1min\) of ([\d-]+ [\d:]+)"

    components = re.findall(component_pattern, scoring_points)
    reasons = re.findall(reason_pattern, scoring_points)
    times = re.findall(time_pattern, scoring_points)

    # Number of root causes depends on the longest of components/reasons/times
    scoringpoints_length = max(len(components),len(reasons),len(times))
    socres_num = len(components)+len(reasons)+len(times)

    def time_difference(time1_str,time2_str):
        from datetime import datetime
        time_format = "%Y-%m-%d %H:%M:%S"
        
        # Clean strings, remove leading/trailing spaces
        time1_str = time1_str.strip()
        time2_str = time2_str.strip()
        
        try:
            time1 = datetime.strptime(time1_str, time_format)
            time2 = datetime.strptime(time2_str, time_format)
        except ValueError:
            print(f"Time format error: '{time1_str}' or '{time2_str}'")
            return False
        
        time_difference = abs(time1 - time2)
        if time_difference.total_seconds() <= 300:
            return True
        else:
            return False

    # Original scoring logic remains unchanged
    scores_get = 0
    passing_criteria = []
    failing_criteria = []

    # Since calculating top3, no need to match event count
    # if scoringpoints_length == prediction_length:  
    best_sore = -1
    for perm in itertools.permutations(predict_results):
        current_score = 0
        current_passing = []
        for i in range(scoringpoints_length):
            if i >= len(perm):
                continue  # Skip indices beyond prediction result length
                
            if len(components) == scoringpoints_length:
                if 'root cause component' in perm[i] and perm[i]['root cause component'] and perm[i]['root cause component'] == components[i]:
                    current_score +=1
                    current_passing.append(components[i])
            if len(reasons) == scoringpoints_length:
                if 'root cause reason' in perm[i] and perm[i]['root cause reason'] and perm[i]['root cause reason'] == reasons[i]:
                    current_score +=1
                    current_passing.append(reasons[i])
            if len(times) == scoringpoints_length:
                if 'root cause occurrence datetime' in perm[i] and perm[i]['root cause occurrence datetime'] and time_difference(times[i], perm[i]['root cause occurrence datetime']):
                    current_score +=1
                    current_passing.append(times[i])
        if current_score>best_sore:
            best_sore = current_score
            passing_criteria = current_passing
    scores_get = best_sore            
    
    failing_criteria = list(set(components+reasons+times)-set(passing_criteria))
    
    final_score = scores_get/socres_num
    bin_score = round(final_score,2)
    
    # Add logic for new top1/top3 evaluation
    # Check if root cause matches completely
    def check_rc_match(pred_rc, gt_components, gt_reasons, gt_times, rc_idx):
        """Check if predicted root cause completely matches the rc_idx-th ground truth root cause"""
        matches = True
        
        # Check component match
        if rc_idx < len(gt_components):
            if 'root cause component' not in pred_rc or not pred_rc['root cause component']:
                matches = False
            elif pred_rc['root cause component'] != gt_components[rc_idx]:
                matches = False
                
        # Check reason match
        if rc_idx < len(gt_reasons):
            if 'root cause reason' not in pred_rc or not pred_rc['root cause reason']:
                matches = False
            elif pred_rc['root cause reason'] != gt_reasons[rc_idx]:
                matches = False
                
        # Check time match
        if rc_idx < len(gt_times):
            if 'root cause occurrence datetime' not in pred_rc or not pred_rc['root cause occurrence datetime']:
                matches = False
            elif not time_difference(gt_times[rc_idx], pred_rc['root cause occurrence datetime']):
                matches = False
                
        return matches
    
    # Check if root cause partially matches (at least one field matches)
    def check_rc_partial_match(pred_rc, gt_components, gt_reasons, gt_times, rc_idx):
        """Check if predicted root cause partially matches the rc_idx-th ground truth root cause (at least one field matches)"""
        has_match = False
        
        # Check component match
        if rc_idx < len(gt_components):
            if 'root cause component' in pred_rc and pred_rc['root cause component'] and pred_rc['root cause component'] == gt_components[rc_idx]:
                has_match = True
                
        # Check reason match
        if rc_idx < len(gt_reasons) and not has_match:
            if 'root cause reason' in pred_rc and pred_rc['root cause reason'] and pred_rc['root cause reason'] == gt_reasons[rc_idx]:
                has_match = True
                
        # Check time match
        if rc_idx < len(gt_times) and not has_match:
            if 'root cause occurrence datetime' in pred_rc and pred_rc['root cause occurrence datetime']:
                if time_difference(gt_times[rc_idx], pred_rc['root cause occurrence datetime']):
                    has_match = True
                
        return has_match
    
    # Calculate top1 and top3 complete and partial matches
    is_top1_match = False  # Complete match
    is_top3_match = False  # Complete match
    is_top1_partial = False  # Partial match
    is_top3_partial = False  # Partial match
    
    # Get prediction count
    pred_count = len(predict_results)
    
    # For case with only 1 ground truth root cause
    if scoringpoints_length == 1:
        # Check if first prediction completely matches
        if pred_count > 0 and check_rc_match(predict_results[0], components, reasons, times, 0):
            is_top1_match = True
            is_top1_partial = True
            is_top3_match = True
            is_top3_partial = True
        else:
            # Check if first prediction partially matches
            if pred_count > 0 and check_rc_partial_match(predict_results[0], components, reasons, times, 0):
                is_top1_partial = True
                is_top3_partial = True
            
            # Check if any prediction completely matches
            for i in range(pred_count):
                if check_rc_match(predict_results[i], components, reasons, times, 0):
                    is_top3_match = True
                    is_top3_partial = True
                    break
            
            # If no complete match but has partial match
            if not is_top3_match:
                for i in range(pred_count):
                    if check_rc_partial_match(predict_results[i], components, reasons, times, 0):
                        is_top3_partial = True
                        break
    
    # For case with 2 ground truth root causes
    elif scoringpoints_length == 2:
        # Check prediction complete matches
        matches = []
        for gt_idx in range(2):
            for pred_idx in range(pred_count):
                if check_rc_match(predict_results[pred_idx], components, reasons, times, gt_idx):
                    matches.append((gt_idx, pred_idx))
        
        # Calculate matched ground truth root cause count
        matched_gt_indices = set([m[0] for m in matches])
        
        # If prediction exactly matches both ground truth root causes, it's a complete match
        if len(matched_gt_indices) == 2:
            is_top1_match = True
            is_top1_partial = True
            is_top3_match = True
            is_top3_partial = True
        # If prediction only matches one ground truth root cause, it's a partial match
        elif len(matched_gt_indices) == 1:
            is_top1_partial = True
            is_top3_partial = True
            
            # Check if all predictions can match all root causes
            all_gt_matched = set(matched_gt_indices)
            for gt_idx in range(2):
                if gt_idx not in all_gt_matched:
                    for pred_idx in range(pred_count):
                        if check_rc_match(predict_results[pred_idx], components, reasons, times, gt_idx):
                            all_gt_matched.add(gt_idx)
                            break
            
            # If all predictions match all ground truth root causes, it's a top3 complete match
            if len(all_gt_matched) == 2:
                is_top3_match = True
        else:
            # If prediction doesn't completely match any ground truth root cause, check for partial matches
            partial_matches = []
            for gt_idx in range(2):
                for pred_idx in range(pred_count):
                    if check_rc_partial_match(predict_results[pred_idx], components, reasons, times, gt_idx):
                        partial_matches.append((gt_idx, pred_idx))
            
            # Calculate partially matched ground truth root cause count
            partial_matched_gt_indices = set([m[0] for m in partial_matches])
            
            # If prediction partially matches any ground truth root cause, it's a top1 partial match
            if len(partial_matched_gt_indices) > 0:
                is_top1_partial = True
            
            # Check all predictions
            all_gt_matched = set()
            partial_gt_matched = set()
            for gt_idx in range(2):
                for pred_idx in range(pred_count):
                    if check_rc_match(predict_results[pred_idx], components, reasons, times, gt_idx):
                        all_gt_matched.add(gt_idx)
                        break
                
                # If no complete match, check for partial match
                if gt_idx not in all_gt_matched:
                    for pred_idx in range(pred_count):
                        if check_rc_partial_match(predict_results[pred_idx], components, reasons, times, gt_idx):
                            partial_gt_matched.add(gt_idx)
                            break
            
            # Determine if complete or partial match based on matched ground truth root cause count
            if len(all_gt_matched) == 2:
                is_top3_match = True
                is_top3_partial = True
            elif len(all_gt_matched) + len(partial_gt_matched) > 0:
                is_top3_partial = True
    
    return passing_criteria, failing_criteria, bin_score, is_top1_match, is_top3_match, is_top1_partial, is_top3_partial


def file_evaluate(prediction_file:str, query_file:str, report_file:str):
    """
    Evaluate a prediction file of certain dataset with corresponding query file and save the evaluation results to a csv file
        args:
            prediction_file: str, the path of the prediction file (csv, with at least one fields: 'prediction')
            query_file: str, the path of a specific dataset recorded labels (csv)
            report_file: str, the path of the evaluation file (csv)
    """ 
    import pandas as pd

    pred_df = pd.read_csv(prediction_file)
    query_df = pd.read_csv(query_file)
    eval_df = pd.DataFrame(columns=[
        "query", "answer", "groundtruth", "passed", "failed", "score", 
        "is_top1_match", "is_top3_match", "is_top1_partial", "is_top3_partial", "task_index"
    ])

    if len(pred_df) != len(query_df):
        raise ValueError("The length of prediction file and record file should be the same")

    for idx in range(len(pred_df)):
        prediction = pred_df.loc[idx, "prediction"]
        scoring_points = query_df.loc[idx, "scoring_points"]
        passing_criteria, failing_criteria, score, is_top1_match, is_top3_match, is_top1_partial, is_top3_partial = evaluate(prediction, scoring_points)
        instruction = query_df.loc[idx, "instruction"]
        task_index = query_df.loc[idx, "task_index"]
        new_row = pd.DataFrame({
            "query": [instruction], 
            "answer": [prediction], 
            "groundtruth": [scoring_points], 
            "passed": [passing_criteria], 
            "failed": [failing_criteria], 
            "score": [score], 
            "is_top1_match": [is_top1_match],
            "is_top3_match": [is_top3_match],
            "is_top1_partial": [is_top1_partial],
            "is_top3_partial": [is_top3_partial],
            "task_index": [task_index]
        })
        eval_df = pd.concat([eval_df, new_row], ignore_index=True)


    if os.path.exists(report_file):
        eval_df.to_csv(report_file, mode='a', header=False, index=False)
    else:
        if not os.path.exists(os.path.dirname(report_file)):
            os.makedirs(os.path.dirname(report_file))
        eval_df.to_csv(report_file, index=False)


def report(report_file):
    """
    Visualize the final result of a report after evaluation
        args:
            report_file: str, report after evaluation
    """
    import pandas as pd

    scores = {
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }
    top1_scores = {
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }
    top3_scores = {
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }
    top1_partial_scores = {
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }
    top3_partial_scores = {
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }
    nums = {
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }
    partial_scores = {
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }
    # Add total score statistics
    total_score_sum = {
        "easy": 0,
        "middle": 0,
        "hard": 0,
    }

    df = pd.read_csv(report_file)
    # By default, task_1-3 is easy, task_4-6 is middle, task_7 is hard. For DIY task specifications, you should change this line to modify the difficulty:
    df["difficulty"] = df["task_index"].apply(lambda x: "easy" if int(x.split('_')[1]) <= 3 else "middle" if int(x.split('_')[1]) <= 6 else "hard")
    scores['easy'] += len(df[(df["score"]==1.0) & (df["difficulty"]=="easy")])
    scores['middle'] += len(df[(df["score"]==1.0) & (df["difficulty"]=="middle")])
    scores['hard'] += len(df[(df["score"]==1.0) & (df["difficulty"]=="hard")])
    
    # Calculate total sum
    total_score_sum['easy'] += df[df["difficulty"]=="easy"]["score"].sum()
    total_score_sum['middle'] += df[df["difficulty"]=="middle"]["score"].sum()
    total_score_sum['hard'] += df[df["difficulty"]=="hard"]["score"].sum()
    
    # Add top1 and top3 statistics
    top1_scores['easy'] += len(df[(df["is_top1_match"]==True) & (df["difficulty"]=="easy")])
    top1_scores['middle'] += len(df[(df["is_top1_match"]==True) & (df["difficulty"]=="middle")])
    top1_scores['hard'] += len(df[(df["is_top1_match"]==True) & (df["difficulty"]=="hard")])
    
    top3_scores['easy'] += len(df[(df["is_top3_match"]==True) & (df["difficulty"]=="easy")])
    top3_scores['middle'] += len(df[(df["is_top3_match"]==True) & (df["difficulty"]=="middle")])
    top3_scores['hard'] += len(df[(df["is_top3_match"]==True) & (df["difficulty"]=="hard")])
    
    # Add top1 and top3 partial match statistics
    top1_partial_scores['easy'] += len(df[(df["is_top1_partial"]==True) & (df["difficulty"]=="easy")])
    top1_partial_scores['middle'] += len(df[(df["is_top1_partial"]==True) & (df["difficulty"]=="middle")])
    top1_partial_scores['hard'] += len(df[(df["is_top1_partial"]==True) & (df["difficulty"]=="hard")])
    
    top3_partial_scores['easy'] += len(df[(df["is_top3_partial"]==True) & (df["difficulty"]=="easy")])
    top3_partial_scores['middle'] += len(df[(df["is_top3_partial"]==True) & (df["difficulty"]=="middle")])
    top3_partial_scores['hard'] += len(df[(df["is_top3_partial"]==True) & (df["difficulty"]=="hard")])
    
    partial_scores['easy'] += len(df[(df["score"]>0) & (df["difficulty"]=="easy")])
    partial_scores['middle'] += len(df[(df["score"]>0) & (df["difficulty"]=="middle")])
    partial_scores['hard'] += len(df[(df["score"]>0) & (df["difficulty"]=="hard")])
    nums['easy'] += len(df[df["difficulty"]=="easy"])
    nums['middle'] += len(df[df["difficulty"]=="middle"])
    nums['hard'] += len(df[df["difficulty"]=="hard"])

    # Original scoring results
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    print(f"{'Original scoring':<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    print(f"{'Class':<12}{'Total(#)':<12}{'Correct(#)':<12}{'Partial(#)':<12}{'Accuracy(%)':<12}{'Partial(%)':<12}{'Avg Score':<12}{'Total Score':<12}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    for key in scores.keys():
        accuracy = scores[key] / nums[key] if nums[key] > 0 else 0
        partial = partial_scores.get(key, 0) if key in partial_scores else "-"
        partial_ratio = partial_scores[key] / nums[key] if nums[key] > 0 else 0
        avg_score = total_score_sum[key] / nums[key] if nums[key] > 0 else 0
        print(f"{key:<12}{nums[key]:<12}{scores[key]:<12}{partial:<12}{accuracy:.2%}{' ':<8}{partial_ratio:.2%}{' ':<8}{avg_score:.4f}{' ':<8}{total_score_sum[key]:.4f}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    total_accuracy = sum(scores.values()) / sum(nums.values()) if sum(nums.values()) > 0 else 0
    total_partial = sum(partial_scores.values())
    total_partial_ratio = total_partial / sum(nums.values()) if sum(nums.values()) > 0 else 0
    total_avg_score = sum(total_score_sum.values()) / sum(nums.values()) if sum(nums.values()) > 0 else 0
    total_all_score = sum(total_score_sum.values())
    print(f"{'Total':<12}{sum(nums.values()):<12}{sum(scores.values()):<12}{total_partial:<12}{total_accuracy:.2%}{' ':<8}{total_partial_ratio:.2%}{' ':<8}{total_avg_score:.4f}{' ':<8}{total_all_score:.4f}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    
    # TOP-1 scoring results
    print(f"\n{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    print(f"{'TOP-1 scoring':<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    print(f"{'Class':<12}{'Total(#)':<12}{'Correct(#)':<12}{'Partial(#)':<12}{'Acc(%)':<12}{'Partial(%)':<12}{'Avg Score':<12}{'Total Score':<12}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    for key in top1_scores.keys():
        accuracy = top1_scores[key] / nums[key] if nums[key] > 0 else 0
        partial_accuracy = top1_partial_scores[key] / nums[key] if nums[key] > 0 else 0
        avg_score = total_score_sum[key] / nums[key] if nums[key] > 0 else 0
        print(f"{key:<12}{nums[key]:<12}{top1_scores[key]:<12}{top1_partial_scores[key]:<12}{accuracy:.2%}{' ':<8}{partial_accuracy:.2%}{' ':<8}{avg_score:.4f}{' ':<8}{total_score_sum[key]:.4f}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    total_top1_accuracy = sum(top1_scores.values()) / sum(nums.values()) if sum(nums.values()) > 0 else 0
    total_top1_partial = sum(top1_partial_scores.values()) / sum(nums.values()) if sum(nums.values()) > 0 else 0
    total_avg_score = sum(total_score_sum.values()) / sum(nums.values()) if sum(nums.values()) > 0 else 0
    total_all_score = sum(total_score_sum.values())
    print(f"{'Total':<12}{sum(nums.values()):<12}{sum(top1_scores.values()):<12}{sum(top1_partial_scores.values()):<12}{total_top1_accuracy:.2%}{' ':<8}{total_top1_partial:.2%}{' ':<8}{total_avg_score:.4f}{' ':<8}{total_all_score:.4f}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    
    # TOP-3 scoring results
    print(f"\n{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    print(f"{'TOP-3 scoring':<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    print(f"{'Class':<12}{'Total(#)':<12}{'Correct(#)':<12}{'Partial(#)':<12}{'Acc(%)':<12}{'Partial(%)':<12}{'Avg Score':<12}{'Total Score':<12}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    for key in top3_scores.keys():
        accuracy = top3_scores[key] / nums[key] if nums[key] > 0 else 0
        partial_accuracy = top3_partial_scores[key] / nums[key] if nums[key] > 0 else 0
        avg_score = total_score_sum[key] / nums[key] if nums[key] > 0 else 0
        print(f"{key:<12}{nums[key]:<12}{top3_scores[key]:<12}{top3_partial_scores[key]:<12}{accuracy:.2%}{' ':<8}{partial_accuracy:.2%}{' ':<8}{avg_score:.4f}{' ':<8}{total_score_sum[key]:.4f}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")
    total_top3_accuracy = sum(top3_scores.values()) / sum(nums.values()) if sum(nums.values()) > 0 else 0
    total_top3_partial = sum(top3_partial_scores.values()) / sum(nums.values()) if sum(nums.values()) > 0 else 0
    total_avg_score = sum(total_score_sum.values()) / sum(nums.values()) if sum(nums.values()) > 0 else 0
    total_all_score = sum(total_score_sum.values())
    print(f"{'Total':<12}{sum(nums.values()):<12}{sum(top3_scores.values()):<12}{sum(top3_partial_scores.values()):<12}{total_top3_accuracy:.2%}{' ':<8}{total_top3_partial:.2%}{' ':<8}{total_avg_score:.4f}{' ':<8}{total_all_score:.4f}")
    print(f"{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}{'-'*12:<12}")

def kpi_evaluate(prediction_file:str, reason:str):
    """
    Evaluate if prediction file contains specific KPI metrics related to given reason
        args:
            prediction_file: str, path to prediction file (.log file)
            reason: str, specific reason to match KPI metrics
    """
    import re
    
    # KPI mapping relationships
    kpi_mapping = {
        # Bank
        "high CPU usage": ["OSLinux-CPU_CPU_CPUCpuUtil"],
        "high memory usage": ["OSLinux-OSLinux_MEMORY_MEMORY_NoCacheMemPerc"],
        "network latency": ["OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT"],
        "network packet loss": ["OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT"],
        "high disk I/O read usage": ["OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKReadWrite"],
        "high disk space usage": ["OSLinux-OSLinux_FILESYSTEM_-tomcat_FSCapacity", "OSLinux-OSLinux_FILESYSTEM_-apache_FSCapacity"],
        "high JVM CPU load": ["JVM-Operating System_7779_JVM_JVM_CPULoad", "JVM-Operating System_7778_JVM_JVM_CPULoad"],
        "JVM Out of Memory (OOM) Heap": ["JVM-Memory_7778_JVM_Memory_NoHeapMemoryUsed", "JVM-Memory_7779_JVM_Memory_NoHeapMemoryUsed"],
        # Telecom
        "CPU fault": ["container_cpu_used", "container_mem_used"],
        "db connection limit": ["Proc_User_Used_Pct", "Login_Per_Sec"],
        "db close": ["On_Off_State"],
        "network delay": ["Sent_queue", "Received_queue"],
        "network loss": ["Sent_queue", "Received_queue"],
        # Market
        "container CPU load": ["container_cpu_usage_seconds"],
        "container memory load": ["container_memory_usage_MB"],
        "container process termination": ["container_threads"],
        "container read I/O load": ["container_fs_reads./dev/vda"],
        "container write I/O load": ["container_fs_writes./dev/vda"],
        "node memory consumption": ["system.mem.used", "system.mem.free"],
        "node disk read I/O consumption": ["system.io.r_await"],
        "node disk write I/O consumption": ["system.io.w_await"],
        "node disk space consumption": ["system.io.avg_q_sz"],
        # Uncertain
        "node CPU load": ["system.cpu.pct_usage", "system.load.1"],
        "node CPU spike": ["system.cpu.user"],
        "container network packet retransmission": ["container_network_receive_MB.eth0"],
        "container network latency": ["container_sockets"],
        "container packet loss": ["container_network_receive_MB.eth0", "container_network_receive_packets.eth0"]
    }
    
    # Check if given reason is in mapping
    if reason not in kpi_mapping:
        print(f"Error: No KPI metrics found for reason '{reason}'")
        return {
            'reason': reason,
            'expected_kpis': [],
            'found_kpis': [],
            'match_count': 0
        }
    
    # Get expected KPI metrics for this reason
    expected_kpis = kpi_mapping[reason]
    
    # Read prediction file content
    with open(prediction_file, 'r', encoding='utf-8') as f:
        prediction_text = f.read()
    
    # Extract execution result section
    result_pattern = r'======generate_tool execution_result=======(.*?)======generate_tool execution_result======='
    execution_results = re.findall(result_pattern, prediction_text, re.DOTALL)
    
    # Store found KPI metrics
    found_kpis = []
    
    for execution_result in execution_results:
        # Directly match KPI metrics
        for kpi in expected_kpis:
            if kpi in execution_result and kpi not in found_kpis:
                found_kpis.append(kpi)
    
    # Calculate match results
    match_count = len(found_kpis)
    total_expected = len(expected_kpis)
    match_rate = match_count / total_expected if total_expected > 0 else 0
    
    # Print results
    print(f"KPI Match Results - Reason: {reason}")
    print(f"{'-'*50}")
    print(f"Expected KPI metrics: {', '.join(expected_kpis)}")
    print(f"Found KPI metrics: {', '.join(found_kpis)}")
    print(f"Match count: {match_count}/{total_expected} ({match_rate:.2%})")
    print(f"{'-'*50}")
    
    # Return results
    return {
        'reason': reason,
        'expected_kpis': expected_kpis,
        'found_kpis': found_kpis,
        'match_count': match_count,
        'total_expected': total_expected,
        'match_rate': match_rate
    }


if __name__ == '__main__':
    """
    Evaluate a list of prediction files with corresponding query files, save the evaluation results, and display the statistic results.
        args:
            p: list, a list of prediction files to evaluate
            q: list, a list of query files to evaluate
            r: str, report file to save
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, nargs='+', help="a list of prediction files to evaluate")
    parser.add_argument("-q", type=str, nargs='+', help="a list of query files to evaluate")
    parser.add_argument("-r", type=str, help="evaluation file to save")
    args = parser.parse_args()

    if len(args.p) != len(args.q):
        raise ValueError("The length of prediction files, query files and evaluation files should be the same")
    if os.path.exists(args.r):
        os.remove(args.r)
    
    for i in range(len(args.p)):
        try:
            file_evaluate(args.p[i], args.q[i], args.r)
        except Exception as e:
            print(f"Error when evaluating the file {args.p[i]}: {e}")
            continue
    
    report(args.r)
    
