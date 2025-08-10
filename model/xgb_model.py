# model/xgb_model.py

import numpy as np
import xgboost as xgb
from sklearn.exceptions import NotFittedError

class XGBModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=10,  # تعداد کمتر برای هر راند فدرال
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        # متغیر برای نگهداری وضعیت آموزش مدل
        self._is_fitted = False

    def get_parameters(self):
        """
        پارامترهای مدل را به صورت باینری (byte array) برمی‌گرداند.
        اگر مدل هنوز آموزش ندیده، یک مقدار اولیه خالی برمی‌گرداند.
        """
        if not self._is_fitted:
            # یک مقدار اولیه و بی‌اثر برای راند اول برمی‌گردانیم
            return [b"initial_params"]

        # پارامترها را به صورت باینری ذخیره می‌کنیم
        booster = self.model.get_booster()
        return [booster.save_raw()]

    def set_parameters(self, parameters):
        """
        پارامترهای دریافتی از سرور را روی مدل اعمال می‌کند.
        """
        params_bytes = parameters[0]
        if not params_bytes or params_bytes == b"initial_params":
            # اگر پارامترها اولیه هستند، کاری انجام نمی‌دهیم
            return

        # یک booster جدید می‌سازیم و پارامترهای دریافتی را در آن بارگذاری می‌کنیم
        booster = xgb.Booster()
        booster.load_model(bytearray(params_bytes))
        self.model._Booster = booster
        self._is_fitted = True

    def fit(self, X_train, y_train):
        """
        مدل را با داده‌های محلی آموزش می‌دهد.
        اگر مدل از قبل پارامتر داشته باشد، آموزش را ادامه می‌دهد (incremental training).
        """
        if self._is_fitted:
            # ادامه آموزش از مدل قبلی (دریافتی از سرور)
            self.model.fit(X_train, y_train, xgb_model=self.model.get_booster())
        else:
            # آموزش برای اولین بار
            self.model.fit(X_train, y_train)
        self._is_fitted = True