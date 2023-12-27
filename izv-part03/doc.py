#!/usr/bin/python3.10
# coding=utf-8

# Author: Lukas Vecerka (xvecer30)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression


def convert_two_digit_year_to_four_digit(year_str, cutoff="23"):
    if year_str is np.nan:
        return np.nan

    if year_str == "XX":
        return np.nan

    if year_str >= cutoff:
        century = "19"
    else:
        century = "20"
    return int(century + year_str)


car_brands_map = {
    1: "Alfa Romeo",
    2: "Audi",
    3: "Avia",
    4: "BMW",
    5: "Chevrolet",
    6: "Chrysler",
    7: "Citroën",
    8: "Dacia",
    9: "Daewoo",
    10: "DAF",
    11: "Dodge",
    12: "Fiat",
    13: "Ford",
    14: "GAZ",
    15: "Ferrari",
    16: "Honda",
    17: "Hyundai",
    18: "IFA",
    19: "IVECO",
    20: "Jaguar",
    21: "Jeep",
    22: "Lancia",
    23: "Land Rover",
    25: "Mazda",
    26: "Mercedes",
    27: "Mitshubishi",
    28: "Moskvič",
    29: "Nissan",
    30: "Oltcit",
    31: "Opel",
    32: "Peugeot",
    33: "Porsche",
    34: "Praga",
    35: "Renault",
    36: "Rover",
    37: "Saab",
    38: "Seat",
    39: "Skoda",
    40: "Scania",
    41: "Subaru",
    42: "Suzuki",
    43: "Tatra",
    44: "Toyota",
    45: "Trabant",
    46: "Vaz",
    47: "Volkswagen",
    48: "Volvo",
    49: "Wartburg",
    50: "Zastava",
    51: "AGM",
    52: "Aro",
    53: "Austin",
    54: "Barkas",
    55: "Daihatsu",
    56: "Datsun",
    57: "Destacar",
    58: "Isuzu",
    59: "Karosa",
    60: "Kia",
    61: "Lublin",
    62: "MAN",
    63: "Maserati",
    64: "Multicar",
    65: "Pontiac",
    68: "SsangYong",
    69: "Talbot",
    70: "Taz",
    71: "Zaz",
    98: "Jiné vyrobené v ČR",
    99: "Jiné vyrobené v zahraničí",
}

tech_issue_type_map = {
    601: "Závada řízení",
    602: "Závada provozní brzdy",
    603: "Závada parkovací brzdy",
    604: "Opotřebení běhounu pláště",
    605: "Defekt pneumatiky (průraz, únik vzduchu)",
    606: "Závada osvětlení",
    607: "Nepřipojená/poškozená hadice pro brzdovou soustavu přívěsu",
    608: "Nesprávné uložení nákladu",
    609: "Upadnutí, ztráta kola vozidla",
    610: "Zablokování kol v důsledku mechanické závady",
    611: "Lom závěsu kola",
    612: "Nezajištěná/poškozená bočnice",
    613: "Závada závěsu pro přívěs",
    614: "Utržená spojovací hřídel",
    615: "Jiná technická závada",
}

# Load the data
df = pd.read_pickle("accidents.pkl.gz")
df.set_index("p1", inplace=True)

# Filter the car accidents only
cars_df = df.copy()
cars_df = cars_df[cars_df["p44"].isin([3, 4])]

# Create a column for technical issues
cars_df["technical_issue"] = cars_df["p10"] == 7

# Create a column for car age
cars_df.dropna(subset=["p47"], inplace=True)
cars_df["car_year"] = cars_df["p47"].apply(
    convert_two_digit_year_to_four_digit
)
cars_df["accident_year"] = cars_df["date"].dt.year
cars_df["car_age"] = cars_df["accident_year"] - cars_df["car_year"]

# Filter the accidents caused by technical issues
tech_acc = cars_df[cars_df["technical_issue"].isin([True])]

# Print the results
print(f"Celkovy pocet nehod osobnich automobilu: {len(cars_df)}")
print(f"Celkovy pocet nehod zpusobenych technickou zavadou: {len(tech_acc)}")
print(
    f"Procento nehod zpusobenych technickou zavadou:\
 {round(len(tech_acc) / len(cars_df) * 100, 2)}%"
)
print(f"Prumerna skoda zpusobena na vozidle v Kc: {cars_df['p53'].mean()}")
print(
    f"Prumerne stari vozidla pri nehode zpusobene technickou zavadou:\
 {round(tech_acc['car_age'].mean(), 2)}"
)

# Plotting the distribution of accidents by car brand
accidents_cnt_by_brand = cars_df["p45a"].value_counts().head(10)
accidents_cnt_by_brand = accidents_cnt_by_brand.rename(car_brands_map)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    x=accidents_cnt_by_brand.index,
    y=accidents_cnt_by_brand.values,
    ax=ax
)

plt.ylabel("Počet nehod")
plt.xlabel("Značka vozidla")
plt.tight_layout()
fig.savefig("accidents_by_brand.png", dpi=300)


# Plotting the distribution of technical issues category
def custom_autopct(pct):
    return "{:.1f}%".format(pct) if pct > 3 else ""


tech_issues_by_type = tech_acc["p12"].value_counts()
tech_issues_by_type = tech_issues_by_type.rename(tech_issue_type_map)

sizes = tech_issues_by_type.values
labels = tech_issues_by_type.index


fig2 = plt.figure(figsize=(10, 6))
wedges, text1, text2 = plt.pie(
    sizes,
    labels=None,
    autopct=custom_autopct,
    startangle=90,
    colors=sns.color_palette("hls", len(sizes)),
)

plt.legend(
    wedges,
    labels,
    title="Technické závady",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
)
plt.savefig("tech_issues_by_type.png", bbox_inches="tight")
plt.close(fig2)


# Plotting the distribution of technical issues by car age
fig3 = plt.figure(figsize=(10, 6))
sns.histplot(data=tech_acc, x="car_age", kde=True, bins=20)

plt.xlabel("Stáří vozidla (let)")
plt.ylabel("Počet technických závad")
plt.savefig("tech_acc.png", dpi=300, bbox_inches="tight")
plt.close(fig3)

# Logistic regression for confirming the hypothesis

cars_df_copy = cars_df.copy()

cars_df_copy = cars_df_copy.dropna(subset=["car_age"])

X = cars_df_copy['car_age'].values.reshape(-1, 1)
Y = cars_df_copy["technical_issue"].astype(int)

logit_model = LogisticRegression()

logit_model.fit(X, Y)

result = logit_model.coef_[0][0]
print(f"Koeficient: {result}")
if result > 0:
    print(
        "Koeficient je vetsi nez 0, takze existuje zavislost\
 mezi stari vozidla a technickymi zavadami."
    )
else:
    print(
        "Koeficient je mensi nez 0, takze neexistuje zavislost mezi\
 stari vozidla a technickymi zavadami."
    )

# if result.pvalues["car_age"] < 0.05:
#     print(
#         "P-hodnota je mensi nez 0.05, takze existuje statisticky\
#  vyznamna zavislost mezi stari vozidla a technickymi zavadami."
#     )
# else:
#     print(
#         "P-hodnota je vetsi nez 0.05, takze neexistuje statisticky\
#  vyznamna zavislost mezi stari vozidla a technickymi zavadami."
#     )

# Print the table

car_idx = cars_df["p45a"].value_counts().head(10)
tech_issues_by_brand = tech_acc["p45a"].value_counts().reindex(car_idx.index)

mean_damage_costs_by_brand = (
    cars_df.groupby("p45a")["p53"].mean().reindex(car_idx.index)
)
mean_damage_costs_by_brand = mean_damage_costs_by_brand.rename(car_brands_map)

table_data = pd.DataFrame(
    {
        "Znacka": accidents_cnt_by_brand.index,
        "Pocet nehod": accidents_cnt_by_brand.values,
        "Podil na nehodach": (
            accidents_cnt_by_brand.values / len(cars_df) * 100
        ),
        "Pocet technickych zavad": tech_issues_by_brand.values,
        "Spolehlivost": (tech_issues_by_brand / car_idx * 100).values,
        "Prumerna skoda": mean_damage_costs_by_brand.values,
    }
)

table_data.set_index("Znacka", inplace=True)
table_data["Podil na nehodach"] = table_data["Podil na nehodach"].apply(
    lambda x: f"{round(x, 2)}%"
)
table_data["Spolehlivost"] = table_data["Spolehlivost"].apply(
    lambda x: f"{round(x, 2)}%"
)
table_data["Prumerna skoda"] = table_data["Prumerna skoda"].apply(
    lambda x: f"{round(x, 2)} Kc"
)

print(table_data)
