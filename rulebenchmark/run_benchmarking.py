import os
import time
import warnings
from os.path import exists

import numpy as np
import pandas as pd

import xgboost as xgb
from corels import CorelsClassifier

from rulelearn.algorithms.rbm.rbm import FeatureBinarizer, FeatureBinarizerFromTrees
from rulelearn.algorithms.r2n.r2n_algo import R2Nalgo
from rulelearn.algorithms.rbm.boolean_rule_cg import BooleanRuleCG as BRCG
from rulelearn.algorithms.ripper import RipperExplainer

from sklearn.tree import DecisionTreeClassifier, export_text

from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from rulebenchmark.data_configs import CONFIG_DICT, core_suite, synth_imbalance_suite, synth_noise_suite, \
    synth_disjunctive_complexity_suite, synth_conjunctive_complexity_suite, synth_linear_complexity_suite, \
    synth_size_suite, test_suite
from rulebenchmark.dtree_helper import get_rules

# Configure a Benchmarking run here
MAX_COMPUTATIONS = 10  # run max. this number of new computations (= pipeline on data set),
# run multiple such batches to get all results
SPLIT_SEED = 42
# SUITES = [core_suite, synth_imbalance_suite, synth_noise_suite, synth_disjunctive_complexity_suite,
#           synth_conjunctive_complexity_suite, synth_linear_complexity_suite, synth_size_suite]
# SUITES = [core_suite]
SUITES = [test_suite]
PIPELINES = [('LENC', 'XGB'),
             ('LENC', 'CART:4'),
             ('LENC', 'CART:6'),
             ('LENC', 'CART:*'),
             ('TREES', 'BRCG'),
             ('NATIVE', 'RIPPER'),
             ('NATIVE', 'RIPPER2'),
             ('NATIVE', 'RIPPER3'),
             ('TREES', 'CORELS'),
             ('TREES', 'BRCG2'),
             ('TREES2', 'BRCG2'),
             ('TREES2', 'BRCG'),
             ('QUANTILE', 'BRCG'),
             ('NATIVE', 'R2N'),
             ('QUANTILE', 'BRCG2'),
             ('QUANTILE', 'RIPPER'),
             ('TREES', 'RIPPER'),
             ('QUANTILE', 'CORELS'),
             ('TREES', 'CORELS2'),
             ('TREES2', 'CORELS2')
             ]


def excluded(binari, pipeline, data_set):
    # exclusion of some configurations which had excessive memory or runtime use in the past
    if data_set.split('_')[0] == 'orange' and (binari == 'QUANTILE' or pipeline == 'R2N'):
        return True
    if data_set.startswith('synth_linear_') and binari == 'TREES2':
        return True
    if binari == 'TREES' and pipeline == 'RIPPER' and data_set in ['german_credit', 'fraud_detection', 'telco_churn']:
        return True
    if binari == 'QUANTILE' and data_set in ['synth_conj_100', 'synth_conj_200', 'synth_conj_400']:
        return True
    return False


RESULT_FILE = 'results_{}.csv'.format(SPLIT_SEED)  # path local to working dir
RUNTIME_EXCEPTION_VAL = -1.0
BINARIZED_DF_OVERFLOW = 2000  # experience on when binarized data frame is too large for subsequent rule induction (on my hardware)
TRAIN_TEST_SPLIT = 0.3
STRING_IMPUTE_VAL = 'missing'  # Imputation value for strings
FLOAT_IMPUTE_VAL = np.finfo(np.float32).min  # Imputation value for floats


# For some rule induction algorithms the pos class is always value 1
def convert(char):
    if char == CONFIG['POS_CLASS']:
        return 1
    else:
        return 0


def load_data(conf):
    loaded_df = pd.read_csv(conf['DATA_SET'], dtype=conf['DATA_TYPES'])
    loaded_df = loaded_df.drop(columns=conf['DROP'])
    print('  Read', len(loaded_df), 'rows from', conf['DATA_SET'])
    loaded_df = loaded_df.dropna(axis=1, how='all')  # drop all completely empty columns
    return loaded_df


def retrieve_result_from_list(data_set_name, results):
    for res in results:
        if res['data_set'] == data_set_name:
            return res
    return None


script_start_time = time.time()
print('Job started for maximal {} computations at {} in directory {}.'.format(MAX_COMPUTATIONS,
                                                                              time.ctime(), os.getcwd()))

new_result_list = []
computation_counter = 0  # counts the number of trainings done

if exists(RESULT_FILE):
    result_df = pd.read_csv(RESULT_FILE)
    print('Read {} lines from {}.'.format(len(result_df), RESULT_FILE))
    result_list = result_df.to_dict('records')
else:
    print('No previous results. Starting from scratch.')
    result_list = []

configs = []
for suite in SUITES:
    configs.extend(suite())

for index, config in enumerate(configs):
    # Load configuration
    CONFIG = CONFIG_DICT[config]
    print('Data set {} of {}: {}'.format(index + 1, len(configs), config))
    data_set_is_loaded = False

    # Check whether data set is already in result file:
    result = retrieve_result_from_list(config, result_list)
    if result is None:
        # Register new data set
        assert CONFIG['TYPE'] == 'BINARY', 'Only binary classification supported.'

        result = {'data_set': config}
        df_from_file = load_data(CONFIG)
        data_set_is_loaded = True

        # Calculate data set specific metrics
        nof_rows = len(df_from_file)
        result['nof_rows'] = nof_rows
        result['nof_col'] = len(df_from_file.columns)
        df_features = df_from_file.drop(columns=[CONFIG['TARGET_LABEL']])

        assert len(df_features.select_dtypes(include=['int64']).columns) == 0, \
            'Integer feature detected: Declare in configuration either as object (categorical) or float (numerical).'

        result['nof_num_features'] = len(df_features.select_dtypes(include=['float64']).columns)
        result['nof_cat_features'] = len(df_features.select_dtypes(include=['object']).columns)

        assert result['nof_num_features'] + result['nof_cat_features'] + 1 == result['nof_col'], \
            'Data frame contains unrecognised data types.'

        target_distribution = df_from_file[CONFIG['TARGET_LABEL']].value_counts()
        nof_pos = target_distribution.loc[CONFIG['POS_CLASS']]
        result['nof_pos'] = nof_pos
        result['pos_ratio'] = round(nof_pos/nof_rows, 4)
        result['use_case'] = CONFIG['META_DATA']['use_case']
        result['origin'] = CONFIG['META_DATA']['origin']

    for (bina, algo) in PIPELINES:
        if (computation_counter < MAX_COMPUTATIONS) and not excluded(bina, algo, config):
            exception = False
            # Configure pipeline
            prefix = bina + '-' + algo
            print('  ', prefix, 'on', config, end=': ')

            # Check whether results are precomputed
            test_col = prefix + '_runtime'
            if (test_col not in result) or (pd.isna(result[test_col])):
                computation_counter += 1
                print('Computation {} of {}'.format(computation_counter, MAX_COMPUTATIONS))

                if not data_set_is_loaded:
                    df_from_file = load_data(CONFIG)
                    data_set_is_loaded = True

                df = df_from_file.copy()  # at least RIPPER messes with the df, hence we draw a fresh copy

                # Prep data A: normalizing label for specific algorithms
                if algo in ('BRCG', 'BRCG2', 'XGB', 'CORELS', 'CORELS2', 'R2N', 'CART:4', 'CART:6', 'CART:*'):
                    df[CONFIG['TARGET_LABEL']] = df[CONFIG['TARGET_LABEL']].map(convert)
                    POS_CLASS = 1
                else:
                    POS_CLASS = CONFIG['POS_CLASS']

                # Prep data B: Imputation whenever needed
                for column in df.select_dtypes(include=['object']).columns:
                    df[column] = df[column].fillna(STRING_IMPUTE_VAL)  # normalize missing string values
                if algo in ['CART:4', 'CART:6', 'CART:*'] or bina in ['TREES', 'TREES2', 'QUANTILE']:
                    for column in df.select_dtypes(include=['float64']).columns:
                        df[column] = df[column].fillna(FLOAT_IMPUTE_VAL)

                # Prep data C: Split into training and test set
                x_train, x_test, y_train, y_test = train_test_split(
                    df.drop(columns=[CONFIG['TARGET_LABEL']]),
                    df[CONFIG['TARGET_LABEL']],
                    test_size=TRAIN_TEST_SPLIT,
                    random_state=SPLIT_SEED)  # use the same training/test subsets for every pipeline

                print('Training', prefix, 'on', config, 'at', time.ctime())
                start_time = time.time()

                # Part 1: Run Binarizer / Encoding
                binarizer_exception = False
                if bina == 'LENC':
                    x_train_bin = x_train
                    x_test_bin = x_test
                    categorical_features = x_train_bin.select_dtypes(include=['object']).columns
                    for col in categorical_features:
                        label_encoder = LabelEncoder()
                        label_encoder = label_encoder.fit(df[col])
                        x_train_bin[col] = label_encoder.transform(x_train_bin[col])
                        x_test_bin[col] = label_encoder.transform(x_test_bin[col])
                elif bina == 'TREES':
                    binarizer = FeatureBinarizerFromTrees(negations=True)  # randomState=42
                    # defaults: treeNum=1, treeDepth=4
                    binarizer = binarizer.fit(x_train, y_train)
                    x_train_bin = binarizer.transform(x_train)
                    x_test_bin = binarizer.transform(x_test)
                elif bina == 'TREES2':
                    binarizer = FeatureBinarizerFromTrees(treeDepth=5, negations=True)  # , randomState=42
                    binarizer = binarizer.fit(x_train, y_train)
                    x_train_bin = binarizer.transform(x_train)
                    x_test_bin = binarizer.transform(x_test)
                elif bina == 'QUANTILE':
                    try:
                        binarizer = FeatureBinarizer(numThresh=9, negations=True)  # , randomState=42
                        binarizer = binarizer.fit(x_train)
                        x_train_bin = binarizer.transform(x_train)
                        x_test_bin = binarizer.transform(x_test)
                    except Exception:
                        result[prefix + 'exception'] = 'binarizer'
                        binarizer_exception = True
                elif bina == 'NATIVE':
                    x_train_bin = x_train
                    x_test_bin = x_test
                if not binarizer_exception:
                    bin_width = len(x_train_bin.columns)
                    result[bina + '_nof_bin_cols'] = bin_width
                    print('Size of binarized data frame', bin_width)
                    if bin_width >= BINARIZED_DF_OVERFLOW:
                        print('binarized data frame too large - interrupting.')
                        result[prefix + 'exception'] = 'binarization_overflow'
                        result[prefix + '_runtime'] = RUNTIME_EXCEPTION_VAL
                        binarizer_exception = True
                        exception = True
                else:
                    print('Binarization failed.')
                    result[prefix + '_runtime'] = RUNTIME_EXCEPTION_VAL
                    exception = True

                # Part 2: Adapter: Binarizer -> Rule Induction
                if not exception:
                    if bina in ['TREES', 'QUANTILE'] and algo in ['RIPPER', 'RIPPER2']:
                        # RIPPER cannot process multi-index produced by these binarizers, hence flatten multi-index
                        x_train_bin = pd.DataFrame(x_train_bin.to_records())
                        x_test_bin = pd.DataFrame(x_test_bin.to_records())
                        x_train_bin = x_train_bin.drop("index", axis=1)
                        x_test_bin = x_test_bin.drop("index", axis=1)
                        x_train_bin.columns = pd.Index(np.arange(1, len(x_train_bin.columns) + 1).astype(str))
                        x_test_bin.columns = pd.Index(np.arange(1, len(x_test_bin.columns) + 1).astype(str))

                # Part 3: Run Rule Induction
                training_exception = False

                if not exception:
                    if algo == 'XGB':
                        estimator = xgb.XGBClassifier(use_label_encoder=False, verbosity=0)
                        estimator.fit(x_train_bin, y_train)
                    elif algo == 'CART:4':
                        estimator = DecisionTreeClassifier(max_depth=4)
                        estimator.fit(x_train_bin, y_train)
                    elif algo == 'CART:6':
                        estimator = DecisionTreeClassifier(max_depth=6)
                        estimator.fit(x_train_bin, y_train)
                    elif algo == 'CART:*':
                        estimator = DecisionTreeClassifier()
                        estimator.fit(x_train_bin, y_train)
                    elif algo == 'RIPPER':
                        estimator = RipperExplainer()  # was: Ripper() d=64 by default random_state=42
                        try:
                            estimator.fit(x_train_bin, y_train, target_label=POS_CLASS)
                        except Exception:
                            result[prefix + 'exception'] = 'training'
                            training_exception = True
                    elif algo == 'RIPPER2':
                        estimator = RipperExplainer(d=128)
                        try:
                            estimator.fit(x_train_bin, y_train, target_label=POS_CLASS)
                        except Exception:
                            result[prefix + 'exception'] = 'training'
                            training_exception = True
                    elif algo == 'RIPPER3':
                        estimator = RipperExplainer(d=256)
                        try:
                            estimator.fit(x_train_bin, y_train, target_label=POS_CLASS)
                        except Exception:
                            result[prefix + 'exception'] = 'training'
                            training_exception = True
                    elif algo == 'BRCG':
                        estimator = BRCG(silent=True)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                estimator.fit(x_train_bin, y_train)
                            except Exception:
                                result[prefix + 'exception'] = 'training'
                                training_exception = True
                    elif algo == 'BRCG2':
                        estimator = BRCG(silent=True, lambda0=.0001, lambda1=.0001)  # default lambda0=.001, lambda1=.001
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                estimator.fit(x_train_bin, y_train)
                            except Exception:
                                result[prefix + 'exception'] = 'training'
                                training_exception = True
                    elif algo == 'CORELS':
                        # exec('estimator = CorelsClassifier(verbosity=[], max_card=2)')
                        estimator = CorelsClassifier(verbosity=[])   # max_card=2 by default
                        # see https://pycorels.readthedocs.io/en/latest/CorelsClassifier.html
                        estimator.fit(x_train_bin, y_train, prediction_name=CONFIG["TARGET_LABEL"])
                    elif algo == 'CORELS2':
                        # exec('estimator = CorelsClassifier(verbosity=[], max_card=2)')
                        cardinality = min(3, bin_width)
                        estimator = CorelsClassifier(verbosity=[], c=0.001, max_card=cardinality)
                        estimator.fit(x_train_bin, y_train, prediction_name=CONFIG["TARGET_LABEL"])
                    elif algo == 'R2N':
                        estimator = R2Nalgo(n_seeds=2,  # 1
                                            min_temp=10 ** -4,
                                            decay_rate=0.98,  # 0.98, 0.998
                                            coef=5 * 10 ** -4,
                                            normalize_num=True,
                                            negation=False)
                        try:
                            estimator.fit(x_train_bin, pd.DataFrame(y_train))
                        except Exception:
                            result[prefix + 'exception'] = 'training'
                            training_exception = True

                    end_time = time.time()
                    training_time = round(end_time-start_time, 2)

                    if training_exception:
                        print('Training failed.')
                        result[prefix + '_runtime'] = RUNTIME_EXCEPTION_VAL
                        print(pd.isna(result[prefix + '_runtime']))
                        exception = True
                    else:
                        print('     Finished training successfully in {} seconds.'.format('%.2f' % training_time))
                        result[prefix + '_runtime'] = training_time

                # Part 4: Evaluation
                prediction_exception = False
                if not exception:
                    try:
                        y_predicted = estimator.predict(x_test_bin)
                    except Exception:
                        print('\t Exception in prediction')
                        result[prefix + 'exception'] = 'prediction'
                        prediction_exception = True
                        exception = True

                if not exception:
                    result[prefix + '_acc'] = round(accuracy_score(y_test, y_predicted), 2)
                    result[prefix + '_adj_bal_acc'] = round(balanced_accuracy_score(y_test, y_predicted, adjusted=True), 2)
                    result[prefix + '_recall'] = round(recall_score(y_test, y_predicted, pos_label=POS_CLASS), 2)
                    result[prefix + '_precision'] = round(precision_score(y_test, y_predicted, pos_label=POS_CLASS,
                                                                          zero_division=1), 2)
                    result[prefix + '_f2'] = round(fbeta_score(y_test, y_predicted, pos_label=POS_CLASS, beta=2,
                                                               zero_division=0), 2)
                    result[prefix + '_f'] = round(fbeta_score(y_test, y_predicted, pos_label=POS_CLASS, beta=1,
                                                              zero_division=0), 2)
                else:
                    result[prefix + '_adj_bal_acc'] = np.nan
                    result[prefix + '_f2'] = np.nan
                    result[prefix + '_acc'] = np.nan
                    result[prefix + '_recall'] = np.nan
                    result[prefix + '_precision'] = np.nan

                export_exception = False
                if not exception:
                    if algo in ('RIPPER', 'RIPPER2', 'RIPPER3', 'BRCG', 'BRCG2', 'R2N'):
                        if algo in ('RIPPER', 'RIPPER2', 'RIPPER3', 'BRCG', 'BRCG2'):
                            rule_set = estimator.explain()
                        else:
                            try:
                                rule_set = estimator.export_rules_to_trxf_dnf_ruleset()
                            except Exception:
                                print('\t Exception in trxf export.')
                                result[prefix + 'exception'] = 'export'
                                export_exception = True
                                exception = True
                        if not export_exception:
                            text_file = open('rules/'+prefix+config+'.txt', "w")
                            _ = text_file.write(str(rule_set))
                            text_file.close()
                            conjunctions = rule_set.list_conjunctions()
                            nof_rules = len(conjunctions)
                            conjunction_len = [len(conjunction) for conjunction in conjunctions]
                            preds_sum = sum(conjunction_len)
                            preds_max = max(conjunction_len, default=0)
                            if nof_rules == 0:
                                preds_avg = 0
                            else:
                                preds_avg = round(preds_sum / nof_rules, 2)

                    elif algo in ['CART:4', 'CART:6', 'CART:*']:
                        try:
                            nof_rules = 0
                            preds_sum = 0
                            preds_max = 0
                            text_file = open('rules/'+prefix+config+'.txt', 'w')
                            for rule in get_rules(estimator, x_train_bin.columns.tolist(), [0, 1]):
                                if rule.find('then class: 1') >= 0:
                                    _ = text_file.write(rule+'\n')
                                    nof_rules += 1
                                    preds = len(rule.split(' and '))
                                    preds_sum += preds
                                    preds_max = max(preds_max, preds)
                            text_file.write('\n\n\n')
                            _ = text_file.write(export_text(estimator, feature_names=x_train_bin.columns.tolist()))
                            text_file.close()
                            preds_avg = round(preds_sum / nof_rules, 2)
                        except Exception:
                            print('\t Exception in decision tree rules export.')
                            result[prefix + 'exception'] = 'export'
                            export_exception = True
                            exception = True

                    elif algo in ['CORELS', 'CORELS2']:
                        rule_list = estimator.rl().rules
                        text_file = open('rules/'+prefix+config+'.txt', 'w')
                        _ = text_file.write(str(rule_list))
                        text_file.close()
                        # nof_rules = len(rule_list)
                        preds_sum = 0
                        preds_max = 0
                        nof_rules = 0
                        stack = []
                        # Converting rule list into DNF rule set
                        for r in rule_list:
                            conjunction_len = len(r['antecedents'])
                            if r['prediction']:
                                factor = np.prod(stack)
                                nof_rules += factor
                                len_rule = len(stack)+conjunction_len
                                preds_sum += factor * len_rule
                                preds_max = max(preds_max, len_rule)
                            else:  # r['prediction'] == False:
                                stack.append(conjunction_len)
                        if nof_rules > 0:
                            preds_avg = round(preds_sum / nof_rules, 2)
                        else:
                            preds_avg = 0

                if algo in ('RIPPER', 'RIPPER2', 'RIPPER3', 'BRCG', 'BRCG2', 'R2N', 'CORELS', 'CORELS2', 'CART:4',
                            'CART:6', 'CART:*') and not exception:
                    result[prefix + '_nof_rules'] = nof_rules
                    result[prefix + '_sum_preds'] = preds_sum
                    result[prefix + '_max_preds'] = preds_max
                    result[prefix + '_avg_preds'] = preds_avg

            else:
                print('Results precomputed.')

    new_result_list.append(result)

if computation_counter == MAX_COMPUTATIONS:
    print('Reached maximal number of {} computations.'.format(MAX_COMPUTATIONS))

result_df = pd.DataFrame(new_result_list)
result_df.to_csv(RESULT_FILE, sep=',', index=False)

script_end_time = time.time()
script_time = script_end_time - script_start_time
print('Job finished at {} in {} seconds.'.format(time.ctime(), '%.2f' % script_time))
