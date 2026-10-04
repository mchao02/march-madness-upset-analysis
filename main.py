import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



def load_data():
    """Load and filter datasets to 2008"""
    tourney = pd.read_csv("data/MNCAATourneyCompactResults.csv")
    seeds = pd.read_csv("data/MNCAATourneySeeds.csv")
    detailed = pd.read_csv("data/MRegularSeasonDetailedResults.csv")

    tourney = tourney[tourney["Season"] >= 2008]
    seeds = seeds[seeds["Season"] >= 2008]
    detailed = detailed[detailed["Season"] >= 2008]

    return tourney, seeds, detailed


def compute_advanced_stats(detailed):
    """Filter through columns, rename, merge, and calculate advanced metrics"""
    wins = detailed[["Season", "WTeamID", "WScore", "WFGM", "WFGA", "WFGM3", "WFTA", "WOR", "WDR", "WTO"]]
    wins.columns = ["Season", "TeamID", "Points", "FGM", "FGA", "FGM3", "FTA", "OR", "DR", "TO"]
    wins_opp = detailed[["LScore", "LFGM", "LFGA", "LFGM3", "LFTA", "LOR", "LDR", "LTO"]]
    wins_opp.columns = ["OppPoints", "OppFGM", "OppFGA", "OppFGM3", "OppFTA", "OppOR", "OppDR", "OppTO"]
    wins_full = pd.concat([wins, wins_opp], axis=1)

    losses = detailed[["Season", "LTeamID", "LScore", "LFGM", "LFGA", "LFGM3", "LFTA", "LOR", "LDR", "LTO"]]
    losses.columns = ["Season", "TeamID", "Points", "FGM", "FGA", "FGM3", "FTA", "OR", "DR", "TO"]
    losses_opp = detailed[["WScore", "WFGM", "WFGA", "WFGM3", "WFTA", "WOR", "WDR", "WTO"]]
    losses_opp.columns = ["OppPoints", "OppFGM", "OppFGA", "OppFGM3", "OppFTA", "OppOR", "OppDR", "OppTO"]
    losses_full = pd.concat([losses, losses_opp], axis=1)

    df = pd.concat([wins_full, losses_full], ignore_index=True)

    df["Poss"] = df["FGA"] - df["OR"] + df["TO"] + 0.475 * df["FTA"]
    df["OppPoss"] = df["OppFGA"] - df["OppOR"] + df["OppTO"] + 0.475 * df["OppFTA"]

    df = df[(df["Poss"] > 0) & (df["OppPoss"] > 0) & (df["FGA"] > 0) & (df["OppFGA"] > 0)]

    df["eFG"] = (df["FGM"] + 0.5 * df["FGM3"]) / df["FGA"]
    df["eFGD"] = (df["OppFGM"] + 0.5 * df["OppFGM3"]) / df["OppFGA"]
    df["TOR"] = df["TO"] / df["Poss"]
    df["TORD"] = df["OppTO"] / df["OppPoss"]
    df["ORB"] = df["OR"] / (df["OR"] + df["OppDR"])
    df["DRB"] = df["DR"] / (df["DR"] + df["OppOR"])
    df["FTR"] = df["FTA"] / df["FGA"]
    df["FTRD"] = df["OppFTA"] / df["OppFGA"]
    df["OffEff"] = (df["Points"] / df["Poss"]) * 100
    df["DefEff"] = (df["OppPoints"] / df["OppPoss"]) * 100
    df["NetRating"] = df["OffEff"] - df["DefEff"]
    df["Tempo"] = (df["Poss"] + df["OppPoss"]) / 2

    important_columns = ["Season", "TeamID", "eFG", "eFGD", "TOR", "TORD","ORB", "DRB", "FTR", "FTRD", "OffEff", "DefEff",
        "NetRating", "Tempo"]

    season_stats = df[important_columns].groupby(["Season", "TeamID"]).mean().reset_index()
    return season_stats


def add_team_stats(tourney, stats):
    """Merge winner and loser season stats into the tournament game matchups"""
    games = tourney.merge(stats,left_on=["Season", "WTeamID"],right_on=["Season", "TeamID"])
    games = games.rename(columns={"eFG": "W_eFG", "eFGD": "W_eFGD", "TOR": "W_TOR", "TORD": "W_TORD", "ORB": "W_ORB",
                                "DRB": "W_DRB", "FTR": "W_FTR", "FTRD": "W_FTRD", "OffEff": "W_OffEff", "DefEff": "W_DefEff", 
                                "NetRating": "W_NetRating", "Tempo": "W_Tempo"})
    games = games.drop(columns=["TeamID"])

    games = games.merge(stats, left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"])
    games = games.rename(columns={"eFG": "L_eFG", "eFGD": "L_eFGD", "TOR": "L_TOR",  "TORD": "L_TORD", "ORB": "L_ORB", 
                                  "DRB": "L_DRB", "FTR": "L_FTR",  "FTRD": "L_FTRD", "OffEff": "L_OffEff", "DefEff": "L_DefEff",
                                  "NetRating": "L_NetRating", "Tempo": "L_Tempo" })
    games = games.drop(columns=["TeamID"])
    return games


def clean_seeds(seeds):
    """Create a seed column"""
    seeds = seeds.copy()
    seeds["SeedNum"] = seeds["Seed"].str[1:3].astype(int)
    return seeds

def add_seeds(games, seeds):
    """Merge winner and loser seeds"""
    games = games.merge(seeds[["Season", "TeamID", "SeedNum"]], left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"])
    games = games.rename(columns={"SeedNum": "WSeed"})
    games = games.drop(columns=["TeamID"])

    games = games.merge(seeds[["Season", "TeamID", "SeedNum"]],left_on=["Season", "LTeamID"],right_on=["Season", "TeamID"])
    games = games.rename(columns={"SeedNum": "LSeed"})
    games = games.drop(columns=["TeamID"])
    return games

def create_features(games):
    """Caluclates the difference feautes (x) and the upset label (y)."""
    games = games.copy()

    games["NetRating_diff"] = games["W_NetRating"] - games["L_NetRating"]
    games["eFG_diff"] = games["W_eFG"] - games["L_eFG"]
    games["TOR_diff"] = games["W_TOR"] - games["L_TOR"]
    games["ORB_diff"] = games["W_ORB"] - games["L_ORB"]
    games["FTR_diff"] = games["W_FTR"] - games["L_FTR"]
    games["Tempo_diff"] = games["W_Tempo"] - games["L_Tempo"]
    games["DefEff_diff"] = games["W_DefEff"] - games["L_DefEff"]
    games["TORD_diff"] = games["W_TORD"] - games["L_TORD"]

    games["upset"] = (games["WSeed"] > games["LSeed"]).astype(int)
    return games

def split_data(games):
    """Splitting the data for testing, training, and validation )"""
    X = games[["NetRating_diff","eFG_diff", "TOR_diff","ORB_diff","FTR_diff", "Tempo_diff", "DefEff_diff", "TORD_diff"]]
    y = games["upset"]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=2500)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size= 0.2, random_state=2500)
    return X_train, X_val, X_test, y_train, y_val, y_test

def feature_selection(X_train, X_val, X_test, selected_features):
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]

    return X_train_selected, X_val_selected, X_test_selected

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model 

def evaluate_model(model, X, y):
    prediction = model.predict(X)
    accuracy = accuracy_score(y, prediction)
    precision = precision_score(y, prediction)
    recall = recall_score(y, prediction)
    f1 = f1_score(y, prediction)
    return prediction, accuracy,precision, recall, f1

def plot_features(model, features):
    """ plots bar graph displaying how much each feature has an impact on the outcome resulting in an upset"""
    coef = model.coef_[0]
    abs_coef = np.abs(coef)

    colors = ["green" if c > 0 else "red" for c in coef]
    plt.figure()
    plt.bar(features, abs_coef,color = colors)
    
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel('Impact (Absolute Value)')

    plt.xticks(rotation=45)
    plt.show()

def plot_class_imbalance(y):
    '''creates bar chart showing the class imbalance'''
    counts = y.value_counts().sort_index()

    labels = [" No Upset", "Upset"]
    plt.figure()
    plt.bar(labels, counts)

    plt.title("Class Imbalance")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()


def create_confusion_matrix(y_true, y_pred, title):
    """Creates confusion matrix"""
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["No Upset", "Upset"])
    display.plot()
    plt.title(title)
    plt.show()



def main():
    tourney, seeds, detailed = load_data()
    stats = compute_advanced_stats(detailed)
    games = add_team_stats(tourney, stats)
    seeds = clean_seeds(seeds)
    games = add_seeds(games, seeds)
    games = create_features(games)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(games)

    model = train_model(X_train, y_train)

    val_pred, val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, X_val, y_val)
    test_pred, test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, X_test, y_test)

    print("Accuracy:", test_acc)
    print("Precision:", test_prec)
    print("Recall:", test_rec)
    print("F1:", test_f1)

    print("Coefficients:",model.coef_ )
    print(games["upset"].value_counts())

    features = ["NetRating_diff","eFG_diff", "TOR_diff","ORB_diff","FTR_diff", "Tempo_diff", "DefEff_diff", "TORD_diff"]
    plot_features(model, features)
    plot_class_imbalance(games["upset"])
    create_confusion_matrix(y_test, test_pred, "Confusion Matrix")

if __name__ == "__main__":
    main()