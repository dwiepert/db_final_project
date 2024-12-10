
#IMPORTS
import pandas as pd
import os
import random
import numpy as np

class Value_Selector(object):
    def __init__(self):
        self.value_selector_history=[]
        self.picked_value=[]
        pass
    def number(self,dataset,percentage):
        number = int((percentage / 100.0) * (len(dataset) ))
        return number
    
    def select_value(self,dataset,number,mute_column):
        for i in range(number):
            random_value = random.randint(1, len(dataset) - 1)
            while random_value in self.value_selector_history:
                random_value = random.randint(1, len(dataset) - 1)
            self.value_selector_history.append(random_value)

            col = random.randint(0, len(dataset.columns) - 1)
            while col in mute_column:
                col = random.randint(0, len(dataset.columns) - 1)

            input_value = dataset.iloc[random_value,col]

            if isinstance(input_value, str):
                try:
                    while (len(input_value) == 0):
                        random_value = random.randint(1, len(dataset) - 1)
                        while random_value in self.value_selector_history:
                            random_value = random.randint(1, len(dataset) - 1)
                        self.value_selector_history.append(random_value)
                        input_value = dataset.iloc[random_value,col]
                except:
                    print(input_value)
            self.picked_value.append([random_value,col,input_value])
        return self.picked_value



class List_selected:
    def __init__(self):
        pass
    def list_selected(self,dataset, percentage,mute_column):
        # create instance from value selector
        instance_value_selector = Value_Selector()

        # how many cell we should change
        columns = dataset.columns.to_list()
        keep_cols = [columns[i] for i in range(len(columns)) if i not in mute_column]

        selected_dataset = dataset[keep_cols]
        #columns = columns[-mute_column]
        number_change = instance_value_selector.number(selected_dataset, percentage)

        # list of the value that picked [[row,col,value]]
        list_selected_value = instance_value_selector.select_value(dataset, number_change,mute_column)

        return list_selected_value,number_change


class Apply_Function(object):
    def __init__(self):
        pass
    def apply_function(self,number_change,list_selected_value,method,dataset):
        print("---------Change according to {} method ---------------\n".format(method.name))

        for i in range(number_change):
            
            #run(row,col,value,dataset)
            result = method.run(list_selected_value[i][0],list_selected_value[i][1],list_selected_value[i][2],dataset)
            dataset.iloc[list_selected_value[i][0],list_selected_value[i][1]] = result
            print("row: {} col: {} : '{}' changed to '{}'  ".format(list_selected_value[i][0], list_selected_value[i][1],list_selected_value[i][2], result))
 
        return dataset
    
class Error_Generator:

    def error_generator(self, method_gen,selector,percentage,dataset,mute_column):

       
        list_selected_value, number_change =selector.list_selected(dataset,percentage,mute_column)

        dataset = Apply_Function.apply_function(self,number_change=number_change, list_selected_value=list_selected_value,method=method_gen, dataset=dataset)
        return dataset

class Implicit_Missing_Value(object):
    def __init__(self,name="Implicit_Missing_Value", dic=None):
        self.name=name
        self.dic={"phone number":"11111111","education":"Some college","Blood Pressurse":"0",
                  "workclass":"?","date":"20010101","Ref_ID":"-1","Regents Num":"s","Junction Control":"-1",
                  "age":"0","Birthday":"20010101","EVENT_DT":"20030101","state":"Alabama","country":"Afghanistan",
                  "email":"...@gmail.com","ssn":"111111111"}
        if dic is not None:

            self.dic=dic


    def run(self,row,col,selected_value,dataset):

        #insted putting the median and mode for implicit missing value
        #we do label matching and acording the dictionary we replace data

        # similar_first=Similar_First()
        # similar_first.similar_first(dataset)
        #
        # mod_value=similar_first.mod_value
        # median_value=similar_first.median_value
        #
        # col_list = [median_value[col], mod_value[col]]
        #
        #
        # rand = np.random.randint(0, 2)
        # selected = col_list[rand]
        #
        # while str(selected_value) == str(selected):
        #     col_list = col_list.remove(selected)
        #     if col_list is None:
        #         selected = median_value + median_value
        #
        # if (isinstance(selected, list)):
        #     if len(selected) > 1:
        #         selected = selected[0]
        if isinstance(selected_value,str):
            ch = list(self.dic.values())
            ch.append(None)
            choice = random.choice(ch)
        else:
            choice = random.choice([None, np.nan])


        return choice

class Outlier_Integer(object):
    def __init__(self,name="Outlier_Integer"):
        self.name=name
    
        
    def run(self,row,col,selected_value,dataset):
        rand = random.randint(0, 3)
        if rand == 0:
            return random.randint(-9999999, 0)
        elif rand == 1:
            return random.randint(-100, 0)
        elif rand == 2:
            return random.randint(200, 300)
        else:
            return random.randint(200, 9999999)
        
def butterfinger(text,prob=0.6,keyboard='querty'):

	keyApprox = {}
	
	if keyboard == "querty":
		keyApprox['q'] = "qwasedzx"
		keyApprox['w'] = "wqesadrfcx"
		keyApprox['e'] = "ewrsfdqazxcvgt"
		keyApprox['r'] = "retdgfwsxcvgt"
		keyApprox['t'] = "tryfhgedcvbnju"
		keyApprox['y'] = "ytugjhrfvbnji"
		keyApprox['u'] = "uyihkjtgbnmlo"
		keyApprox['i'] = "iuojlkyhnmlp"
		keyApprox['o'] = "oipklujm"
		keyApprox['p'] = "plo['ik"

		keyApprox['a'] = "aqszwxwdce"
		keyApprox['s'] = "swxadrfv"
		keyApprox['d'] = "decsfaqgbv"
		keyApprox['f'] = "fdgrvwsxyhn"
		keyApprox['g'] = "gtbfhedcyjn"
		keyApprox['h'] = "hyngjfrvkim"
		keyApprox['j'] = "jhknugtblom"
		keyApprox['k'] = "kjlinyhn"
		keyApprox['l'] = "lokmpujn"

		keyApprox['z'] = "zaxsvde"
		keyApprox['x'] = "xzcsdbvfrewq"
		keyApprox['c'] = "cxvdfzswergb"
		keyApprox['v'] = "vcfbgxdertyn"
		keyApprox['b'] = "bvnghcftyun"
		keyApprox['n'] = "nbmhjvgtuik"
		keyApprox['m'] = "mnkjloik"
		keyApprox[' '] = " "
	else:
		print ("Keyboard not supported.")

	probOfTypoArray = []
	probOfTypo = int(prob * 100)

	buttertext = ""
	for letter in text:
		lcletter = letter.lower()
		if not lcletter in keyApprox.keys():
			newletter = lcletter
		else:
			if random.choice(range(0, 100)) <= probOfTypo:
				newletter = random.choice(keyApprox[lcletter])
			else:
				newletter = lcletter
		# go back to original case
		if not lcletter == letter:
			newletter = newletter.upper()
		buttertext += newletter

	return buttertext


class Typo_Butterfingers(object):
    def __init__(self,name="Typo_Butterfingers",prob=0.6):
        self.name=name
        self.prob=prob


    def run(self,row,col,selected_value,dataset):
        temp = butterfinger(selected_value,prob=self.prob)
        return temp
    
class Random_Domain(object):
    def __init__(self,name="Outlier_Integer"):
        self.name=name
    
        
    def run(self,row,col,selected_value,dataset):

        rand_row = random.randint(0, len(dataset) - 1)
        
        rand_col = random.randint(0, len(dataset.columns) - 1)

        while rand_col == col:
            rand_col = random.randint(0, len(dataset.columns) - 1)

        return str(dataset.iloc[rand_row,rand_col])

class Switch_Relationship(object):
    def __init__(self,possible1, possible2, name="Switch_Relationship"):
        self.name=name
        self.possible1 = possible1
        self.possible2 = possible2
    
        
    def run(self,row,col,selected_value,dataset):
       if col == 47:
           p = [i for i in self.possible1 if i != selected_value]
           return random.choice(p)
       elif col == 48:
           p = [i for i in self.possible2 if i != selected_value]
           return random.choice(p)
       #country = r[]

def clean_players(orig_dir='./original_soccer', save_dir='./new_datasets/players/'):
    #POST CONVERTING TO CSV FROM SQLLITE
    cdf = pd.read_csv(os.path.join(orig_dir, 'country.csv'))
    ldf = pd.read_csv(os.path.join(orig_dir, 'league.csv'))
    mdf = pd.read_csv(os.path.join(orig_dir, 'match.csv'))
    pdf = pd.read_csv(os.path.join(orig_dir, 'player.csv'))
    padf = pd.read_csv(os.path.join(orig_dir, 'player_att.csv'))
    tdf = pd.read_csv(os.path.join(orig_dir, 'team.csv'))
    tadf = pd.read_csv(os.path.join(orig_dir, 'team_att.csv'))

    #rename
    #cdf: country_id name
    cdf = cdf.rename(columns={'id':'country_id', 'name':'country'})
    #ldf: id, country_id, name
    ldf = ldf.rename(columns={'id': 'league_id', 'name':'league'})
    #mdf: id, country_id, league_id
    mdf = mdf.rename(columns={'id':'match_id'})
    #pdf: id, player_api_id, player_fifa_api_id
    pdf = pdf.rename(columns={'id':'player_id'})
    #padf: id, player_api_id, player_fifa_api_id
    padf = padf.rename(columns={'id':'player_attribute_id'})
    #tdf: id, team_api_id, team_fifa_api_id
    tdf = tdf.rename(columns={'id':'team_id'})
    #tadf: id, team_fifa_api_id, teamp_api_id 
    tadf = tadf.rename(columns={'id':'team_attribute_id'})

    # initial merge and filter
    m1 = pd.merge(cdf, ldf)
    m2 = pd.merge(mdf, m1)

    m3 = pd.merge(padf, pdf)
    m4 = pd.merge(tadf, tdf)

    matches = m2.dropna()
    players = m3.dropna()
    teams = m4.dropna()

    # secondary merge (get country and league for players)
    p = matches.T.reset_index().T
    p1 = p.T[55:77].T
    p2 = p.T[115:117].T
    player_matches = pd.concat([p1,p2],axis=1)
    pa2 = players['player_api_id'].to_list()
    uniquepa2 = list(set(pa2))
    save = {}
    for ind, row in player_matches.iterrows():
        for p in row:
            if not isinstance(p, str):
                if p not in save:
                    save[int(p)] = {'country': row[115], 'league': row[116]}
    c = []
    l = []

    for p in uniquepa2:
        if p in save:
            c.append(save[p]['country'])
            l.append(save[p]['league'])
        else:
            c.append(None)
            l.append(None)
    pdf2 = pd.DataFrame({'player_api_id':uniquepa2, 'country': c, 'league': l})
    players = pd.merge(players, pdf2).dropna()
    players = players.sample(frac=1)

    keep_cols = ['player_attribute_id', 'overall_rating', 'preferred_foot', 'attacking_work_rate', 'volleys', 'free_kick_accuracy', 'player_name', 'height', 'weight', 'country', 'league']
    players = players[keep_cols]
    os.makedirs(save_dir, exist_ok=True)
    players.to_csv(os.path.join(save_dir,'players_clean.csv'), index=False)
    return players

def get_mute_cols_str(dataset):
    mute = []
    types = dataset.dtypes
    i = 0
    indexes = types.index.to_list()
    types = types.to_list()
    for i in range(len(indexes)):

        if types[i] != 'O' or indexes[i] == 'date' or indexes[i] == 'birthday':
            mute.append(i)
    return mute

def get_mute_cols_numeric(dataset):
    mute = []
    types = dataset.dtypes
    i = 0
    indexes = types.index.to_list()
    types = types.to_list()
    for i in range(len(indexes)):

        if types[i] == 'O' or i ==0:
            mute.append(i)
    return mute

def create_typos(dataset, mute_column=[0, 1, 9]):
    d = dataset.copy()
    myselector=List_selected()
    mygen=Error_Generator()

    mymethod=Typo_Butterfingers(prob=0.15)
    new_dataset=mygen.error_generator(method_gen=mymethod,selector=myselector,percentage=30,dataset=d,mute_column=mute_column)
    return new_dataset

def create_missing_data(dataset):
    d = dataset.copy()
    myselector=List_selected()
    mygen=Error_Generator()

    mymethod=Implicit_Missing_Value(dic={
                "0":"",
                "1":"null",
                "2":"?",
                "3":"NULL",
                "4":"unknown",
                "5":'""',
                "6":'N/A',
    })

    new_dataset=mygen.error_generator(method_gen=mymethod,selector=myselector,percentage=30,dataset=d,mute_column=[0])
    return new_dataset

def create_outlier(dataset, mute_column=[0, 2, 3, 4, 5, 6, 7, 8, 10, 11]):
    d = dataset.copy()
    myselector=List_selected()
    mygen=Error_Generator()

    mymethod=Outlier_Integer()
    new_dataset=mygen.error_generator(method_gen=mymethod,selector=myselector,percentage=30,dataset=d,mute_column=mute_column)
    return new_dataset

def create_domain(dataset):
    d = dataset.copy()
    myselector=List_selected()
    mygen=Error_Generator()

    mymethod=Random_Domain()
    new_dataset=mygen.error_generator(method_gen=mymethod,selector=myselector,percentage=30,dataset=d,mute_column=[0])
    return new_dataset

### SOCCER SPECIFIC 
def create_fn(dataset, mute_column=[0]):
    d = dataset.copy()
    myselector=List_selected()
    mygen=Error_Generator()
    
    possible1 = list(set(dataset['country'].to_list()))
    possible2 = list(set(dataset['league'].to_list()))
    mymethod=Switch_Relationship(possible1=possible1, possible2=possible2)
    new_dataset=mygen.error_generator(method_gen=mymethod,selector=myselector,percentage=30,dataset=d,mute_column=mute_column)
    return new_dataset


def dirty_players(players, save_dir):
    str_mute_cols = get_mute_cols_str(players)
    typos = create_typos(players, mute_column=str_mute_cols)
    typos.to_csv(os.path.join(save_dir, 'players_typos.csv'),index=False)

    msd = create_missing_data(players)
    msd.to_csv(os.path.join(save_dir, 'players_missing.csv'),index=False)

    num_mute_cols = get_mute_cols_numeric(players)
    od = create_outlier(players, mute_column=num_mute_cols)
    od.to_csv(os.path.join(save_dir, 'players_outliers.csv'),index=False)

    dd = create_domain(players)
    dd.to_csv(os.path.join(save_dir, 'players_domain.csv'),index=False)

    keep_cols = [9,10]
    mute_cols = [i for i in range(len(players.columns)) if i not in keep_cols]
    fd = create_fn(players, mute_column=mute_cols)
    fd.to_csv(os.path.join(save_dir, 'players_fn.csv'),index=False)
    return None 

def main():
    orig_dir = './original_soccer/'
    save_dir = './new_datasets/players/'

    players_clean = clean_players(orig_dir, save_dir)
    dirty_players(players_clean, save_dir)


if __name__ == '__main__':
    main()