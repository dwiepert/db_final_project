#IMPORTS
##built-in
import argparse
import csv
import glob
import os
import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
#requires torch, accelerate

### LLM
class Qwen:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        st = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        et = time.time()
        print(f'{(et-st)/60} s elapsed loading model')

    def __call__(self, prompt):
        messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=400
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

def compare_csv_files(file1_path, file2_path, limit):
    num_differences = 0
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        i = 0
        for line1, line2 in zip(file1, file2):
            if i >= limit: 
                continue
            if line1.strip() != line2.strip():
                num_differences += 1
            i += 1
    return num_differences

class DataFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, mode='r')
        self.csv_reader = csv.reader(self.file)
        self.header = next(self.csv_reader)
        self.make_int = ['overall_rating', 'volleys', 'free_kick_accuracy', 'weight']
        if all([m in self.header] for m in self.make_int):
            self.players=True
            self.change_inds = [i for i in range(len(self.header)) if self.header[i] in self.make_int]
        else:
            self.players = False
    
    def get_next_row(self):
        try:
            row = next(self.csv_reader)
            if self.players:
                for i in self.change_inds:
                    t = row[i]
                    row[i] = t
            return row
        except StopIteration:
            return None
    
    def get_header(self):
        return self.header
    
    def reset(self):
        self.file.close()
        self.file = open(self.file_path, mode='r')
        self.csv_reader = csv.reader(self.file)
        self.header = next(self.csv_reader)

    def __del__(self):
        self.file.close()

class Dataset:
    def __init__(self, dirty_file_path, clean_file_path) -> None:
        self.dirty = DataFile(dirty_file_path)
        self.clean = DataFile(clean_file_path)

    def get_header(self):
        return self.dirty.get_header()

    def get_next_row(self):
        dirty_row = self.dirty.get_next_row()
        clean_row = self.clean.get_next_row()

        if dirty_row is None or clean_row is None:
            return None
        
        error_headers = [h_col for d_col, c_col, h_col in zip(dirty_row, clean_row, self.dirty.get_header()) if d_col != c_col]

        return (dirty_row, clean_row, error_headers)
    

    def reset(self):
        self.dirty.reset()
        self.clean.reset()

class Detector:
    def __init__(self, run_fn, model_type='qwen2.5-72B-instruct', detection_type="column"):
        self.run_fn = run_fn
        self.model_type = model_type
        self.detection_type = detection_type

    def _generate_prompt(self, row, header, dataset_type):
        formated_header = '|'.join(header)
        formatted_row = '|'.join(row)

        prompt = f"I have tabular data with the following columns {formated_header}. " 
        if self.detection_type=='metadata':
            prompt += self._get_metadata(dataset_type)
        prompt += f"Detect if there are any errors, typos, or missing entries in the following row: "
        prompt += f"{formatted_row}. "
        prompt += "Only name and discuss columns with errors. Do NOT mention correct columns."

        return prompt 
    
    def _get_metadata(self, dataset):
        if dataset == 'players':
            return self._get_players_metadata()
    
    def _get_players_metadata(self):
        prompt = "The overall_rating column is a numeric value between 0 and 100. "
        prompt += "The preferred_foot column is either right or left. "
        prompt += "The attacking_work_rate column is one of the following: low, medium, high. "
        prompt += "The volleys column is an numeric value between 0 and 100. "
        prompt += "The free_kick_accuracy column is an numeric value between 0 and 100. "
        prompt += "The player_name column is a value like Aaron Hunt or Stephane M'Bia or Sung-Yeung Ki or Suso or Tulio de Melo or Victor Hugo Montano. "
        prompt += "The height column is a numeric value value between 120.0 and 250.0. "
        prompt += "The weight column is an numeric value value between 100 and 300. "
        prompt += "The country column is one of the following values: England, France, Germany, Italy, Spain. "
        prompt += "The league column is one of the following values: England Premier League, France Ligue 1, Germany 1. Bundesliga, Italy Serie A, Spain LIGA BBVA. "
        prompt += "If the league column is 'England Premier League' then the country must be 'England'. "
        prompt += "If the league column is 'France Ligue 1' then the country must be 'France'. "
        prompt += "If the league column is 'Germany 1. Bundesliga' then the country must be 'Germany'. "
        prompt += "If the league column is 'Italy Serie A' then the country must be 'Italy'. "
        prompt += "If the league column is 'Spain LIGA BBVA' then the country must be 'Spain'."
        #prompt += "Only discuss columns with errors. Do NOT mention correct columns."
        return prompt
    
    def _get_qwen_phrases(self):
        return ['there do not appear to be any errors', 'appears to be correctly formatted', 'there are no errors']
    
    def get_columns_mentioned_in_response(self, res, header):
        header = [col.lower() for col in header]
        res = res.lower()

        cols = [col for col in header if f"'{col}'" in res or f"\"{col}\"" in res or f" {col} " in res or f"*{col}*" in res or f"\n{col} " in res]

        if 'row_id' in cols:
            cols.remove('row_id')

        return cols
    
    def help_get_response(self, row, header, dataset_type="players", printres=True):
        prompt = self._generate_prompt(row, header, dataset_type)

        if 'qwen' in self.model_type:
            phrases = self._get_qwen_phrases()
        else:
            phrases = []
        
        res= self.run_fn(prompt)

        if printres:
            print(res)

        if any([f in res.lower() for f in phrases]):
            errors = []
            #print(res)
        else:
            if 'no error' in res:
                res = res.split('no error')[0]

            errors = self.get_columns_mentioned_in_response(res, header)
            if printres:
                print(errors)
            #if len(errors) > 3:
            #    print(errors)
        return errors
    
class Evaluate:

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.header = dataset.get_header()
    
    # precision: correct error detections / total number of error detections
    # recall: correct error detections / total number of errors
    # f1-score: 2 * (precions * recall) / (precision + recall)

    def eval(self, detector, limit=10000, mod=1000):
        self.dataset.reset()

        is_fn = 'fn.csv' in self.dataset.dirty.file_path
        if is_fn:
            print('TODO: figure out difference for functional dependencies')
        num_reported_errors = 0
        num_correct_reported_errors = 0
        num_errors = 0

        i = 0

        while i < limit:
            r = self.dataset.get_next_row()
            if r is None:
                break

            dirty_row, clean_row, error_columns = r 

            if error_columns != []:
                check = True

            num_errors += len(error_columns)

            if i % mod == 0:
                printres = True
            else:
                printres = False
            if i % 100 == 0:
                st = time.time()
                print(st)
            reported_error_columns = detector.help_get_response(dirty_row, self.header,dataset_type='players', printres=printres)

            if i % 100 == 0:
                et = time.time()
                print(f"{(et-st)/60} minutes elapsed generating response")
            for col in reported_error_columns:
                if is_fn and 'country' in error_columns:
                    if col == 'league':
                        num_errors += 1
                if is_fn and 'league' in error_columns:
                    if col == 'country':
                        num_errors += 1

                if col in error_columns:
                    # correct error found!
                    num_correct_reported_errors += 1
            
            num_reported_errors += len(reported_error_columns)

            if i % 5 == 0:
                print(f"\t\t\t{i}")
                print(f"\t\t\t\t{num_errors, num_reported_errors, num_correct_reported_errors}")

            i += 1

        precision = 0
        if num_reported_errors != 0:
            precision = num_correct_reported_errors / num_reported_errors

        recall = 0
        if num_errors != 0:
            recall = num_correct_reported_errors / num_errors

        f1 = 0
        if precision + recall != 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        scores = {'recall':[recall], 'precision':[precision], 'f1':[f1], 'num_errors': [num_errors], 'num_reported_errors': [num_reported_errors], 'num_correct_reported_errors':[num_correct_reported_errors]}

        return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_ptrn', default='./new_datasets/players/*.csv')
    parser.add_argument('-s', '--save_dir', default='./outputs')
    parser.add_argument('-d','--dataset_name', default='players')
    parser.add_argument('-l', '--limit', type=int, default=5000)
    parser.add_argument('-pt', '--pretrained_path', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument("-m", "--model_name", default="qwen2.5-14b-instruct")
    args = parser.parse_args()

    paths = glob.glob(args.path_ptrn)
    clean_path = [f for f in paths if 'clean' in f][0]
    dirty_paths = [f for f in paths if 'clean' not in f]
    model = Qwen(model_name=args.pretrained_path)
    print('Model loaded')

    output_df = None

    for d in dirty_paths:
        num_differences = compare_csv_files(d, clean_path, limit=args.limit)
        error_type = os.path.splitext(d.split(sep="_")[-1])[0]
        print(f'Num differences between {os.path.basename(d)} and {os.path.basename(clean_path)}: {num_differences}')
        ds = Dataset(dirty_file_path=d, clean_file_path=clean_path)
        col_detector = Detector(model, model_type=args.model_name.lower(), detection_type='column')
        md_detector = Detector(model, model_type=args.model_name.lower(), detection_type='metadata')
        evaluator = Evaluate(ds)
        col_outputs = evaluator.eval(col_detector, limit=args.limit)
        col_outputs['dirty_file'] = [d]
        col_outputs['clean_file'] = [clean_path]
        col_outputs['dataset'] = [args.dataset_name]
        col_outputs['prompt'] = ['column']
        col_outputs['error_type'] = [error_type]
        col_outputs['num_differences'] = [num_differences]
        

        if output_df is None:
            output_df = pd.DataFrame(col_outputs)
        else:
            temp = pd.DataFrame(col_outputs)
            output_df = pd.concat([output_df, temp], axis=0)
        
        
        md_outputs = evaluator.eval(md_detector, limit=args.limit)
        md_outputs['dirty_file'] = [d]
        md_outputs['clean_file'] = [clean_path]
        md_outputs['dataset'] = [args.dataset_name]
        md_outputs['prompt'] = ['metadata']
        md_outputs['error_type'] = [error_type]
        md_outputs['num_differences'] = [num_differences]
        temp = pd.DataFrame(md_outputs)
        output_df = pd.concat([output_df, temp], axis=0)

        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir,f'qwen_{args.dataset_name}_{error_type}.csv')
        output_df.to_csv(save_path, index=False)
        output_df = None
    
    #save_dir = './outputs'
    #os.makedirs(save_dir, exist_ok=True)
    #save_path = save_dir+f'qwen_{args.dataset_name}_{error_type}.csv'
    #output_df.to_csv(save_path, index=False)

    return None

if __name__ == '__main__':
    main()