import pickle

def to_pickle(lis,path):
    """
    This function takes a list and a path as input and saves the list as a pickle file at the specified path.
    Args:
        lis: list - The list to be saved as a pickle file.
        path: str - The path where the pickle file will be saved.
    Returns:
        None
    """
    with open(path+".pkl", 'wb') as f:
        pickle.dump(lis, f)

def from_pickle(path):
    """
    This function takes a path as input and loads the pickle file at the specified path and returns the loaded list.
    Args:
        path: str - The path where the pickle file is located.
    Returns:
        lis: list - The loaded list from the pickle file.
    """
    with open(path,'rb') as f:
        lis=pickle.load(f)
    return lis
if __name__=="__main__":
    #a=from_pickle("Bert.pkl")
    a=from_pickle("DNAbert_6.pkl.pkl")
    for  i in a:
        print(i[0])