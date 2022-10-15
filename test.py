def test(x_test, uniq_sites_alice, uniq_sites_other, model):
    x_test[SITES] = x_test[SITES].fillna(0).astype(int)
    x_test = unique_sites_insert(x_test, uniq_sites_alice, uniq_sites_other)

    x_test = preprocess_time(x_test)
    x_test = pd.concat([x_test, cyclical(x_test[['num_of_month', 'hour', 'day_of_the_week']])], axis=1)
    x_test['session_time'] = normalize(x_test['session_time'])

    x_test = x_test.drop(columns=['day_of_the_week', 'num_of_month', 'hour', 'first_time', 'last_time', 'target'], axis=1)
    x_test = x_test.fillna(0).drop(columns=TIMES, axis=1)

    test_times, test_days = x_test[['session_time', 'morning', 'day', 'evening', 'night',
                                     'num_of_month_sin', 'num_of_month_cos', 'hour_sin', 'hour_cos',
                                     'day_of_the_week_sin', 'day_of_the_week_cos']], \
                              x_test[['unique_sites_alice', 'unique_sites_other', 'weekend']]

    all_test = create_matrix(x_test[SITES], test_days, test_times)

    # todo: dataclass
    return predict(model=model, x_test)
