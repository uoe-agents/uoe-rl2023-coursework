"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""
import pytest
import shlex


@pytest.fixture()
def collect_questions():
    import rl2023.answer_sheet as answer_sheet
    questions = []
    for i in dir(answer_sheet):
        item = getattr(answer_sheet, i)
        if callable(item):
            questions.append(item)
    return questions


@pytest.fixture()
def question_specs():

    # question_types:
    #    - multi_choice : args = (*choices)
    #    - short_answer : args = (max_words)

    # question_name : (question_type, question_type_args)
    specs = {
        "question2_1": ("multi_choice", ("a", "b")),
        "question2_2": ("multi_choice", ("a", "b")),
        "question2_3": ("multi_choice", ("a", "b")),
        "question2_4": ("short_answer", (100,)),
        "question3_1": ("multi_choice", ("a", "b", "c")),
        "question3_2": ("multi_choice", ("a", "b", "c")),
        "question3_3": ("multi_choice", ("a", "b", "c")),
        "question3_4": ("multi_choice", ("a", "b", "c", "d", "e")),
        "question3_5": ("multi_choice", ("a", "b", "c", "d", "e")),
        "question3_6": ("short_answer", (100,)),
        "question3_7": ("short_answer", (150,)),
        "question3_8": ("short_answer", (100,)),
        "question5_1": ("short_answer", (200,)),
        }

    return specs


def test_missing_questions(collect_questions, question_specs):
    question_names = list(question_specs.keys())
    collected_questions = [q.__name__ for q in collect_questions]
    for question in question_names:
        assert question in collected_questions, f"Missing {question} in answer_sheet.py"


def test_extra_questions(collect_questions, question_specs):
    question_names = list(question_specs.keys())
    for question in collect_questions:
        assert question.__name__ in question_names, f"Found {question.__name__} in answer_sheet.py not required for assignment."


def test_missing_answers(collect_questions):
    missing_questions = [question.__name__ for question in collect_questions if len(question()) == 0]
    error_msg = '\n'.join(missing_questions)
    assert not missing_questions, f"Missing answers to questions: \n{error_msg}"


def test_questions_outputs(collect_questions, question_specs):

    def get_specs(question, question_specs):
        return question_specs[question.__name__]

    def test_multi_choice(question, question_type_args):
        #Not case sensitive
        answer = question().lower()
        return answer in question_type_args, f"Answer '{question()}' from {question.__name__} not in {question_type_args}"

    def test_short_answer(question, question_type_args):
        return len(shlex.split(question())) <= question_type_args[0], f"Answer '{question()}' from {question.__name__} exceeds max words"

    errors = []
    for question in collect_questions:
        question_type, question_type_args = get_specs(question, question_specs)
        if question_type == "multi_choice":
            check, error = test_multi_choice(question, question_type_args)
        elif question_type == "short_answer":
            check, error = test_short_answer(question, question_type_args)
        else:
            raise NotImplementedError(f"Question type {question_type} not implemented")
        if not check:
            errors.append(error)
    error_msg = '\n'.join(errors)
    assert not errors, f"Answers to the following questions are not among possible answers: \n{error_msg}"