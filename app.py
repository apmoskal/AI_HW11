@app.post("/predict")
def predict(
    lag1: float = Form(...),
    lag2: float = Form(...),
    lag3: float = Form(...),
    lag4: float = Form(...),
    lag5: float = Form(...),
    lag6: float = Form(...)
):
    data = pd.DataFrame([{
        "Sales_Lag_1_Month": lag1,
        "Sales_Lag_2_Month": lag2,
        "Sales_Lag_3_Month": lag3,
        "Sales_Lag_4_Month": lag4,
        "Sales_Lag_5_Month": lag5,
        "Sales_Lag_6_Month": lag6
    }])
    dmatrix = xgb.DMatrix(data)
    prediction = model.predict(dmatrix)[0]
    return {"prediction": float(prediction)}
    data = np.array(input.sales).reshape(1, -1)
    prediction = model.predict(data)[0]
    return {"prediction": float(prediction)}
