import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset and parse timestamps."""
    df = pd.read_csv(filepath)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def plot_energy_usage(df: pd.DataFrame):
    """Plot energy consumption over time."""
    plt.figure(figsize=(12, 5))
    plt.plot(df["Timestamp"], df["Energy_Consumption_kWh"], label="Energy Consumption (kWh)", color="blue")
    plt.xlabel("Time")
    plt.ylabel("Energy (kWh)")
    plt.title("Energy Consumption Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_battery_vs_energy(df: pd.DataFrame):
    """Plot battery charge vs energy consumption."""
    plt.figure(figsize=(12, 5))
    plt.plot(df["Timestamp"], df["Battery_Charge_%"], label="Battery Charge (%)", color="green")
    plt.plot(df["Timestamp"], df["Energy_Consumption_kWh"], label="Energy Consumption (kWh)", color="blue")
    plt.xlabel("Time")
    plt.ylabel("Battery Charge & Energy (kWh)")
    plt.title("Battery Charge vs Energy Consumption")
    plt.legend()
    plt.grid(True)
    plt.show()


def prepare_features(df: pd.DataFrame):
    """Scale features and return train/test split."""
    X = df[["Temperature_C", "Solar_Intensity", "Battery_Charge_%"]]
    y = df["Energy_Consumption_kWh"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    return train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)


def evaluate_model(model, X_test, y_test, name: str):
    """Evaluate a model and print results."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - Mean Squared Error: {mse:.2f}, R² Score: {r2:.2f}")
    return mse, r2


def main():
    filepath = "data/home_energy_usage.csv" 

    # Load data
    df = load_data(filepath)
    print(df.info())

    # Visualizations
    plot_energy_usage(df)
    plot_battery_vs_energy(df)

    # Prepare training data
    X_train, X_test, y_train, y_test = prepare_features(df)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, "Linear Regression")

    # Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest")


if __name__ == "__main__":
    main()
