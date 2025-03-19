import pandas as pd  # Для работы с данными в табличном виде
import numpy as np  # Для численных операций
import xgboost as xgb  # Библиотека для градиентного бустинга (следуя из презентации вы её используете)
from sklearn.model_selection import TimeSeriesSplit  # Для временного разделения данных
import matplotlib.pyplot as plt  # Для визуализации
import matplotlib.dates as mdates  # Для форматирования дат на графиках
import seaborn as sns  # Улучшенная визуализация
from matplotlib.dates import DateFormatter  # Форматирование дат

# Выбор даты
selected_date = '2025-03-08'  # Дата, для которой требуется сделать прогноз

# Загрузка данных из Excel-файла
df = pd.read_excel('id_точки 16.xlsx')
# Преобразуем столбец с датами в формат datetime для корректной работы с временными рядами
df['OpenDate.Typed'] = pd.to_datetime(df['OpenDate.Typed'])

# Функция для генерации признаков (features)
def create_features(df):
    # Сортируем данные по названию блюда и дате для корректного расчета лагов и скользящих статистик
    df = df.sort_values(['DishName', 'OpenDate.Typed'])
    
    #Временные признаки
    # День недели (0-Пн, 6-Вс) - помогает уловить недельные паттерны
    df['day_of_week'] = df['OpenDate.Typed'].dt.dayofweek
    # Месяц (1-12) - для учета сезонных изменений
    df['month'] = df['OpenDate.Typed'].dt.month
    # День месяца (1-31) - для внутри-месячных паттернов
    df['day_of_month'] = df['OpenDate.Typed'].dt.day
    # Признак выходного дня (True/False) - для учета повышенного спроса в выходные
    df['is_weekend'] = df['OpenDate.Typed'].dt.weekday >= 5
    # Квартал (1-4) - для учета квартальных сезонных изменений
    df['quarter'] = df['OpenDate.Typed'].dt.quarter
    
    # Лаговые признаки (прошлые значения продаж) 
    # Создаем признаки с продажами за 1, 2, 7, 14 и 28 дней до текущей даты
    for lag in [1, 2, 7, 14, 28]:
        df[f'lag_{lag}'] = df.groupby('DishName')['DishAmountInt'].shift(lag).fillna(0)
        # shift(lag) смещает данные на lag дней назад, fillna(0) заполняет пропуски нулями
    
    # Скользящие статистики (усреднение, отклонение, максимум)
    # Для окон 3, 7, 14, 28 и 60 дней рассчитываем:
    for window in [3, 7, 14, 28, 60]:
        # Скользящее среднее - показывает общий тренд
        df[f'rolling_mean_{window}'] = df.groupby('DishName')['DishAmountInt'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        # Скользящее стандартное отклонение - показывает волатильность
        df[f'rolling_std_{window}'] = df.groupby('DishName')['DishAmountInt'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
        )
        # Скользящий максимум - показывает пики спроса
        df[f'rolling_max_{window}'] = df.groupby('DishName')['DishAmountInt'].transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )
    
    # Взаимодействия признаков
    # Произведение дня недели и месяца для учета специфических паттернов
    df['day_month_interaction'] = df['day_of_week'] * df['month']
    # Тренд продаж (разница с предыдущим днем) - показывает рост/падение
    df['sales_trend'] = df.groupby('DishName')['DishAmountInt'].diff().fillna(0)
    
    # Кодирование категориального признака
    # Преобразуем названия блюд в числовой код (модель не работает с текстом)
    df['DishName_encoded'] = df['DishName'].astype('category').cat.codes
    return df

# Подготовка данных
# Группируем данные по дате и блюду, суммируя продажи (минимум 1 запись)
df = df.groupby(['OpenDate.Typed', 'DishName'], as_index=False)['DishAmountInt'].sum(min_count=1)
# Создаем полный список дат от минимальной до максимальной в данных
all_dates = pd.date_range(start=df['OpenDate.Typed'].min(), end=df['OpenDate.Typed'].max(), freq='D')
# Получаем список всех уникальных блюд
all_dishes = df['DishName'].unique()
# Создаем полную таблицу со всеми комбинациями дат и блюд, заполняя пропуски нулями
full_df = df.set_index(['OpenDate.Typed', 'DishName']).reindex(
    pd.MultiIndex.from_product([all_dates, all_dishes], 
    names=['OpenDate.Typed', 'DishName']), fill_value=0).reset_index()

# Генерируем признаки для полного набора данных
full_df = create_features(full_df)

# Разделение на обучающую и тестовую выборки
# Выбираем последние 60 дней для тестовой выборки (последние 2 месяца)
test_dates = np.sort(full_df['OpenDate.Typed'].unique())[-60:]
# Обучающая выборка - все данные, кроме последних 60 дней
train_df = full_df[~full_df['OpenDate.Typed'].isin(test_dates)] # ~ для обратного действия
# Тестовая выборка - последние 60 дней (копируем для сохранения исходных данных)
test_df = full_df[full_df['OpenDate.Typed'].isin(test_dates)].copy()

# Подготовка признаков для модели
# Столбцы, которые не участвуют в обучении (дата, название блюда, целевая переменная)
exclude_cols = ['OpenDate.Typed', 'DishName', 'DishAmountInt']
# Все остальные столбцы - признаки для модели
features = [col for col in full_df.columns if col not in exclude_cols]
# Разделяем данные на признаки (X) и целевую переменную (y)
X_train, y_train = train_df[features], train_df['DishAmountInt']
X_test, y_test = test_df[features], test_df['DishAmountInt']

# Обучение модели XGBoost
model = xgb.XGBRegressor( # Гиперпараметры подобраны с помощью GridSearchCV
    objective='reg:squarederror',  # Регрессия для прогнозирования непрерывной переменной
    n_estimators=400,  # Количество деревьев 
    max_depth=6,  # Максимальная глубина деревьев 
    learning_rate=0.1,  # Скорость обучения 
    subsample=0.9,  # Используем 90% данных для обучения каждого дерева 
    colsample_bytree=0.8,  # Используем 80% признаков для каждого дерева 
    tree_method='hist',  # Метод построения деревьев (hist - быстрый для больших данных)
    random_state=42  # Фиксация случайного seed для воспроизводимости(0/42/NONE)
)
# Обучение модели на обучающих данных
model.fit(X_train, y_train)

# Оценка модели на тестовых данных
# Получаем предсказания модели (clip(0) - не допускаем отрицательных значений)
y_pred = model.predict(X_test).clip(0)
# Маска для исключения нулевых фактических значений при расчете метрик
valid_mask = y_test > 0
y_test_valid, y_pred_valid = y_test[valid_mask], y_pred[valid_mask]

# Расчет метрик качества
mae = np.mean(np.abs(y_pred_valid - y_test_valid))  # Средняя абсолютная ошибка
rmse = np.sqrt(np.mean((y_pred_valid - y_test_valid)**2))  # Среднеквадратичная ошибка
r2 = 1 - (np.sum((y_test_valid - y_pred_valid)**2) / np.sum((y_test_valid - y_test_valid.mean())**2))  # Коэффициент детерминации
mape = np.mean(np.abs((y_test_valid - y_pred_valid)/y_test_valid)) * 100  # Средняя абсолютная процентная ошибка

# Вывод метрик
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, MAPE: {mape:.1f}%")

# Функция для прогнозирования на заданную дату
def make_prediction(date_str):
    date = pd.to_datetime(date_str)
    max_available_date = full_df['OpenDate.Typed'].max()  # Последняя доступная дата в данных
    
    # Создаем DataFrame для прогноза с заданной датой и всеми блюдами
    pred_df = pd.DataFrame({
        'OpenDate.Typed': [date] * len(all_dishes),
        'DishName': all_dishes
    })
    
    # Генерация временных признаков для прогнозной даты
    pred_df['day_of_week'] = date.weekday()
    pred_df['month'] = date.month
    pred_df['day_of_month'] = date.day
    pred_df['is_weekend'] = date.weekday() >= 5
    pred_df['quarter'] = date.quarter
    
    # Лаговые признаки для прогноза
    for lag in [1, 2, 7, 14, 28]:
        # Определяем дату для лага (не позже последней доступной)
        lag_date = min(date - pd.Timedelta(days=lag), max_available_date)
        # Получаем последние доступные продажи для каждого блюда на дату лага
        lag_values = full_df[
            (full_df['OpenDate.Typed'] <= lag_date) & 
            (full_df['DishName'].isin(all_dishes))
        ].groupby('DishName')['DishAmountInt'].last().to_dict()
        # Заполняем лаговые признаки (fillna(0) - если данных нет)
        pred_df[f'lag_{lag}'] = pred_df['DishName'].map(lag_values).fillna(0)
    
    # Скользящие статистики для прогноза
    for window in [3, 7, 14, 28, 60]:
        # Определяем период для расчета статистик (до прогнозной даты)
        window_end = min(date, max_available_date)
        window_start = window_end - pd.Timedelta(days=window-1)
        # Получаем доступные даты в этом периоде
        available_dates = full_df['OpenDate.Typed'].unique()
        valid_dates = pd.date_range(start=window_start, end=window_end, freq='D').intersection(available_dates)
        # Рассчитываем статистики для каждого блюда
        stats = full_df[
            (full_df['OpenDate.Typed'].isin(valid_dates)) & 
            (full_df['DishName'].isin(all_dishes))
        ].groupby('DishName')['DishAmountInt'].agg(['mean', 'std', 'max']).fillna(0)
        # Заполняем признаки скользящих статистик
        for stat in ['mean', 'std', 'max']:
            pred_df[f'rolling_{stat}_{window}'] = pred_df['DishName'].map(stats[stat]).fillna(0)
    
    # Дополнительные признаки
    pred_df['day_month_interaction'] = pred_df['day_of_week'] * pred_df['month']
    pred_df['sales_trend'] = 0  # Для прогноза тренд неизвестен (ставим 0)
    # Кодируем названия блюд в числовой код (берем из обучающих данных)
    pred_df['DishName_encoded'] = pred_df['DishName'].map(full_df.groupby('DishName')['DishName_encoded'].first().to_dict())
    
    # Прогноз
    X_pred = pred_df[features]
    # Предсказываем и округляем (clip(0) - не меньше нуля)
    pred_df['DishAmountInt'] = model.predict(X_pred).clip(0).round()
    # Возвращаем только блюда с прогнозом >0, сортируя по убыванию
    return pred_df[pred_df['DishAmountInt'] > 0].sort_values('DishAmountInt', ascending=False)

# Получение и вывод прогноза
prediction = make_prediction(selected_date)
print(f"\nПрогноз на {selected_date}:")
print(prediction[['DishName', 'DishAmountInt']].to_string(index=False))

# Визуализация результатов
# Создаем DataFrame с фактическими и предсказанными значениями
test_df['Predicted'] = y_pred.round()
comparison_df = test_df[['OpenDate.Typed', 'DishAmountInt', 'Predicted']].copy()

# График агрегированных продаж по дням
# Агрегируем данные по дням (суммируем продажи всех блюд)
daily_comparison = comparison_df.groupby('OpenDate.Typed').sum(numeric_only=True).reset_index()

# Создаем график
fig, ax = plt.subplots(figsize=(14, 7))
# Форматируем ось X (недельные интервалы)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Строим графики фактических и прогнозных значений
ax.plot(daily_comparison['OpenDate.Typed'], daily_comparison['DishAmountInt'], 
        label='Фактические продажи', marker='o', linestyle='-')
ax.plot(daily_comparison['OpenDate.Typed'], daily_comparison['Predicted'], 
        label='Прогноз модели', marker='x', linestyle='--')

# Добавляем заголовок, подписи осей и легенду
plt.title('Сравнение фактических и прогнозных продаж (тестовый период)', fontsize=14)
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Количество блюд', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Добавляем метрики на график
metrics_text = f'MAE: {mae:.1f}\nRMSE: {rmse:.1f}\nMAPE: {mape:.1f}%'
plt.gcf().text(0.92, 0.75, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()

# График для топ-5 блюд
# Создаем DataFrame с фактическими и предсказанными значениями
test_df['Predicted'] = y_pred.round()
comparison_df = test_df[['OpenDate.Typed', 'DishName', 'DishAmountInt', 'Predicted']].copy()

# Выбираем топ-5 блюд по суммарным продажам в тестовом периоде
top_dishes = comparison_df.groupby('DishName')['DishAmountInt'].sum().nlargest(5).index
filtered_comparison = comparison_df[comparison_df['DishName'].isin(top_dishes)]

# Создаем график для каждого блюда
plt.figure(figsize=(16, 10))
sns.set(style="whitegrid")

for i, dish in enumerate(top_dishes, 1):
    plt.subplot(3, 2, i)
    dish_data = filtered_comparison[filtered_comparison['DishName'] == dish]
    
    # Строим фактические и прогнозные значения
    sns.lineplot(x='OpenDate.Typed', y='DishAmountInt', 
                 data=dish_data, marker='o', label='Фактически')
    sns.lineplot(x='OpenDate.Typed', y='Predicted', 
                 data=dish_data, marker='x', linestyle='--', label='Прогноз')
    
    # Форматируем график
    plt.title(f'Прогноз vs Факт: {dish}', fontsize=12)
    plt.xlabel('Дата')
    plt.ylabel('Количество')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    # Формат даты на оси X (месяц-день)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))

plt.tight_layout()
plt.show()
