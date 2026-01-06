
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%%
PARTY_NAMES = []
MUN_NAMES=[]
STAT_NAMES=[]

CONFIG = {
    'test_size': 0.2,           
    'random_seed': 42,          

    'input_size': None,         
    'output_size': 9,          
    'hidden_layers': [64, 32], 
    'activation_fn': nn.ReLU(), 
    
    'batch_size': 16,           
    'epochs': 500,
    'print_every': 50,             
    'learning_rate': 0.001,     

    'dropout_rate': 0.2,       
    'l2_reg': 1e-4           
}
idx_trn=[]
idx_val=[]
scaler = StandardScaler()

#%%
class ElectionMLP(nn.Module):
    def __init__(self, config):
        super(ElectionMLP, self).__init__()
        
        layers = []
        input_dim = config['input_size']
        

        for hidden_dim in config['hidden_layers']:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(config['activation_fn'])
            
            if config['dropout_rate'] > 0:
                layers.append(nn.Dropout(p=config['dropout_rate']))
            
            input_dim = hidden_dim # Nästa lager tar detta lagers utput som input
        
        # Sista lagret (Output layer)
        # Vi använder ingen aktiveringsfunktion här (linear output) för regression,
        # eller Softmax om du vill tvinga summan till 1 (men MSELoss hanterar råa värden bra).
        layers.append(nn.Linear(input_dim, config['output_size']))
        
        # Paketera allt i en sekvens
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x=self.model(x)
        return torch.softmax(x,dim=1)

class MunicipalityDataset(Dataset):
    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preprocess_data():
    print("Laddar data...")

    municipality_data=pd.read_excel('Municipality Data.xlsx')
    X = municipality_data.iloc[:,1:].values

    global MUN_NAMES,STAT_NAMES
    MUN_NAMES=municipality_data.iloc[:,0].values
    STAT_NAMES=municipality_data.columns.to_list()
    
    
    election_res=pd.read_excel('Result Swedish General Election 2022- by Municipality.xlsx')
    y = election_res.iloc[:,1:].values

    global PARTY_NAMES
    PARTY_NAMES=election_res.columns[1:].to_list()

    global idx_trn,idx_val
    idx_trn,idx_val = train_test_split(
        np.arange(len(X)), 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_seed']
    )
    
    global scaler
    X[idx_trn] = scaler.fit_transform(X[idx_trn]) 
    X[idx_val]=scaler.transform(X[idx_val])
    
    
    return X,y

def run_training(X,y):

    CONFIG['input_size'] = X[idx_trn].shape[1]

    # Skapa DataLoaders
    train_dataset = MunicipalityDataset(X[idx_trn], y[idx_trn])
    val_dataset = MunicipalityDataset(X[idx_val], y[idx_val])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

  
    model = ElectionMLP(CONFIG)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=CONFIG['learning_rate'], 
                           weight_decay=CONFIG['l2_reg'])

    print(f"Modell skapad: {CONFIG['hidden_layers']} dolda lager.")
    print("Startar träning...\n")

    for epoch in range(CONFIG['epochs']):
      
        model.train() 
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()           
            predictions = model(X_batch)    
            loss = criterion(predictions, y_batch) 
            loss.backward()                
            optimizer.step()           
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        #Validation
        model.eval() 
        val_loss = 0.0
        
        with torch.no_grad(): 
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)

        #Print result
        if (epoch + 1) % CONFIG['print_every'] == 0:
            print(f"Epok [{epoch+1}/{CONFIG['epochs']}] | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f}")

    print("\nTräning klar!")
    return model

def predict_municipality(index,model,X,y):
    
    model.eval()
    
    exempel_data = torch.tensor(X[idx_val][index:index+1], dtype=torch.float32)
    sann_data = y[idx_val][index]
    
    with torch.no_grad():
        prediktion = model(exempel_data).numpy()[0]
    
   
    df_resultat = pd.DataFrame({
        "Parti": PARTY_NAMES,
        "Prediktion (%)": np.round(prediktion * 100, 2),
        "Faktiskt (%)": np.round(sann_data * 100, 2),
        "Skillnad (p.e.)": np.round((prediktion - sann_data) * 100, 2)
    })
    
    print(f"\n--- Analys för {MUN_NAMES[idx_val][index]}")
    print(df_resultat.to_string(index=False))
    
    return df_resultat

#Experiment stage, does not really work that well
def predict_profiles(index,model):
    profiles=pd.read_excel('Kommundata.xlsx',sheet_name='Väljarprofiler')
    X_prof=profiles.iloc[:,1:].values
    profile_names=profiles.iloc[:,0].values

    X_prof=scaler.transform(X_prof)

    model.eval()
    profile=torch.tensor(X_prof[index:index+1], dtype=torch.float32)

    with torch.no_grad():
        prediktion = model(profile).numpy()[0]

    df_resultat = pd.DataFrame({
        "Parti": PARTY_NAMES,
        "Prediktion (%)": np.round(prediktion * 100, 2),
    })
    
    print(f"\n--- Analys för {profile_names[index]} med index: ---")
    print(df_resultat.to_string(index=False))
    


#%%
if __name__ == "__main__":
    X,y=preprocess_data()
    model=run_training(X,y)
#%%
    predict_municipality(7,model,X,y)
    
#%%
    #predict_profiles(0,model)
