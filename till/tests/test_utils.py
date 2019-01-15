from till import utils


def test_str_join():
    assert utils.str_join(["example", "path"]) == 'example/path'
    assert type(utils.str_join(["example", "path"])) is str


def test_create_folder(): pass


def test_get_from_dict():
    d = {'a': 'Apple', 'b': 'Banana', 'c': 'Carrot'}
    a, b, c = utils.get_from_dict(d, ['a', 'b', 'c'])
    assert a == 'Apple'
    assert b == 'Banana'
    assert c == 'Carrot'
