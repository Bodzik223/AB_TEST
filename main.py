import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
from math import ceil
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# розрахувати ефективний розмір вибірки на основі очікуваного коефіцієнта конверсії
effect_size = sms.proportion_effectsize(0.12, 0.14)

# розрахунок вимог до обсягу вибірки
required_sample_size = sms.NormalIndPower().solve_power(
    effect_size,
    power=0.8,
    alpha = 0.05,
    ratio = 1
)

required_sample_size = ceil(required_sample_size)
print(f'Required Sample Size: {required_sample_size}')

# перегляд наданих даних
df = pd.read_csv('ab_data.csv')
print(pd.crosstab(df['group'],df['landing_page']))

# очищення помилкових даних
df_clean_1 = df[~((df['group'] == 'control') & (df['landing_page'] == 'new_page'))]
df_clean_2 =  df_clean_1[~((df_clean_1['group'] == 'treatment') & (df_clean_1['landing_page'] == 'old_page'))]
print(pd.crosstab(df_clean_2['group'],df_clean_2['landing_page']))
print(df_clean_2.shape)

# перевірка дублікатів сеансів користувачів
user_session_count = df_clean_2['user_id'].value_counts()
multiple_session_users_index = user_session_count[user_session_count > 1].index

# видалення дублікатів сеансів користувачів
df_clean_3 = df_clean_2[~df_clean_2['user_id'].isin(multiple_session_users_index)]
print(df_clean_3.shape)

# створення контрольної та досліджуваної групи вибірки
control_sample_group = df_clean_3[df_clean_3['group'] == 'control'].sample(n = required_sample_size, random_state=22)
treatment_sample_group = df_clean_3[df_clean_3['group'] == 'treatment'].sample(n = required_sample_size, random_state=22)

print(f'Контрольна група: {control_sample_group.shape[0]} \n'
      f'Досліджувана група: {treatment_sample_group.shape[0]}')

ab_test = pd.concat([control_sample_group,treatment_sample_group], axis = 0).reset_index()

print(ab_test.head())
print(f"Група для тесту:\n{ab_test['group'].value_counts()}")

conversion_rate = ab_test.groupby('group')['converted']
conversion_rate = conversion_rate.agg([np.mean, lambda x: np.std(x)])
conversion_rate.columns = ['conversion_rate', 'std_deviation']

print(conversion_rate)

# тестування гіпотез

control_results = ab_test[ab_test['group'] == 'control']['converted']
treatment_results = ab_test[ab_test['group'] == 'treatment']['converted']

success_count = [control_results.sum(),treatment_results.sum()]
nobs = [control_results.count(),treatment_results.count()]

stat, pval = proportions_ztest(count=success_count, nobs=nobs)
(ci_control_low,ci_treatment_low),(ci_control_upp, ci_treatment_upp) =  proportion_confint(count=success_count, nobs=nobs)


print(f'z-статистика : {stat:.5f}')
print(f'p-значення : {pval:.5f}')
print(f'Довірчий інтервал 95% для контрольної групи [{ci_control_low:.5f},{ci_control_upp:.5f}]')
print(f'Довірчий інтервал 95% для досліджуваної групи [{ci_treatment_low:.5f},{ci_treatment_upp:.5f}]')

if pval > 0.05:
    print('p-значення значно перевищує поріг p-значення, ми не можемо відкинути нульову гіпотезу. Новий дизайн не демонструє значущих результатів')
else:
     print(f'p-значення менше/дорівнює пороговому значенню p-значення, ми можемо відхилити нульову гіпотезу. Новий дизайн демонструє значущість')