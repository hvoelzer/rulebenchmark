import rulelearn.trxf.core.conjunction as conjunction
import rulelearn.trxf.core.dnf_ruleset as ruleset
from rulelearn.trxf.classifier.ruleset_classifier import RuleSetClassifier, RuleSelectionMethod
from rulelearn.trxf.core.utils import batch_evaluate
import pandas as pd


def transform(b, pos_value):
    if b:
        return pos_value


class ConstantPosClassifier:

    def __init__(self, pos_value):
        """
        @param pos_value: the constant value that is always returned
        """
        conj = conjunction.Conjunction([])
        self._pos_value = pos_value
        self._rule_set = ruleset.DnfRuleSet([conj], pos_value)

    def predict(self, data_frame):
        return batch_evaluate(self._rule_set, data_frame).apply(transform, pos_value=self._pos_value)


# def constant_pos_classifier(pos_value):
#     conj = conjunction.Conjunction([])
#     rs = ruleset.DnfRuleSet([conj], pos_value)
#     classifier = RuleSetClassifier([rs], rule_selection_method=RuleSelectionMethod.FIRST_HIT, default_label='dummy')
#     return classifier

# def constant_pos_rule_set(pos_value):
#     conj = conjunction.Conjunction([])
#     rule_set = ruleset.DnfRuleSet([conj], pos_value)
#     return rule_set
#
#
# def batch_predict(clf, data_frame):
#     pass

# dict1 = {'key 1': 'value 1', 'key 2': 'value 2', 'key 3': 'value 3'}
# dict2 = {'key 1': 'value 4', 'key 2': 'value 5', 'key 3': 'value 6'}
#
# df = pd.DataFrame([dict1, dict2])
# print(df.info())
# rs = constant_pos_rule_set('hello')
# print(batch_evaluate(rs, df).apply(transform, pos_value='hi'))

# clf = constant_pos_classifier('hello')
# print(clf)
# my_val = {'age': 25, 'estimated_income': 70000}
# result = clf.predict(my_val)
# print(result)
