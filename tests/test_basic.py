import pytest


@pytest.fixture
def lexer():
    from basic import BasicLexer
    return BasicLexer()


@pytest.mark.parametrize(
    'test_string, expected_tokens',
    (
        ('A = 2', (
            ('ID', 'A'),
            ('EQUALS', '='),
            ('NUMBER', 2),
        )),
        ('A = A + 1', (
            ('ID', 'A'),
            ('EQUALS', '='),
            ('ID', 'A'),
            ('PLUS', '+'),
            ('NUMBER', 1),
        )),
        ('PRINT A', (
            ('PRINT', 'PRINT'),
            ('ID', 'A'),
        )),
        ('10 A = 5', (
            ('LINENO', 10),
            ('LINE', 'A = 5'),
        )),
        (' 10 A = 5', (
            ('LINENO', 10),
            ('LINE', 'A = 5'),
        )),
        ('10.2 A = 5', (
            ('LINENO', 10),
            ('LINE', '.2 A = 5'),
        )),
        ('10', (('LINENO', 10),)),
        ('  10', (('LINENO', 10),)),
        ('10  ', (('LINENO', 10),)),
        ('10     A = 5', (
            ('LINENO', 10),
            ('LINE', 'A = 5'),
        )),
        ('LIST', (('LIST', 'LIST'),)),
        ('PRINT 1 2 3', (
            ('PRINT', 'PRINT'),
            ('NUMBER', 1),
            ('NUMBER', 2),
            ('NUMBER', 3),
        )),
        ('PRINT A = 3', (
            ('PRINT', 'PRINT'),
            ('ID', 'A'),
            ('EQUALS', '='),
            ('NUMBER', 3),
        )),
        ('PRINTA=3', (
            ('PRINT', 'PRINT'),
            ('ID', 'A'),
            ('EQUALS', '='),
            ('NUMBER', 3),
        )),
        ('PRINT"A"', (
            ('PRINT', 'PRINT'),
            ('STRING', 'A'),
        )),
        ('PRINT"A"A', (
            ('PRINT', 'PRINT'),
            ('STRING', 'A'),
            ('ID', 'A'),
        )),
        ('PRINT"A"REM', (
            ('PRINT', 'PRINT'),
            ('STRING', 'A'),
            ('REM', 'REM'),
        )),
        ('PRINT"A":REMTHISISIGNORED', (
            ('PRINT', 'PRINT'),
            ('STRING', 'A'),
            ('COLON', ':'),
            ('REM', 'REMTHISISIGNORED'),
        )),
        ('PRINT "A" : REMTHISISIGNORED', (
            ('PRINT', 'PRINT'),
            ('STRING', 'A'),
            ('COLON', ':'),
            ('REM', 'REMTHISISIGNORED'),
        )),
        ('REM THIS IS IGNORED', [('REM', 'REM THIS IS IGNORED')]),
        ('  REM THIS IS IGNORED', [('REM', 'REM THIS IS IGNORED')]),
        ('IF A THEN PRINT "B" ELSE PRINT "C"', (
            ('IF', 'IF'),
            ('ID', 'A'),
            ('THEN', 'THEN'),
            ('PRINT', 'PRINT'),
            ('STRING', 'B'),
            ('ELSE', 'ELSE'),
            ('PRINT', 'PRINT'),
            ('STRING', 'C'),
        )),
        ('IF 0 THEN PRINT "B" ELSE PRINT "C"', (
            ('IF', 'IF'),
            ('NUMBER', 0),
            ('THEN', 'THEN'),
            ('PRINT', 'PRINT'),
            ('STRING', 'B'),
            ('ELSE', 'ELSE'),
            ('PRINT', 'PRINT'),
            ('STRING', 'C'),
        )),
        ('PRINT 3 * 4 + 5', (
            ('PRINT', 'PRINT'),
            ('NUMBER', 3),
            ('MULTIPLY', '*'),
            ('NUMBER', 4),
            ('PLUS', '+'),
            ('NUMBER', 5),
        )),
        ('PRINT 4 * 2 ^ 3', (
            ('PRINT', 'PRINT'),
            ('NUMBER', 4),
            ('MULTIPLY', '*'),
            ('NUMBER', 2),
            ('POWER', '^'),
            ('NUMBER', 3),
        )),  # = 4 * 8 = 32
        ('PRINT 3 + 5 * -2', (
            ('PRINT', 'PRINT'),
            ('NUMBER', 3),
            ('PLUS', '+'),
            ('NUMBER', 5),
            ('MULTIPLY', '*'),
            ('MINUS', '-'),
            ('NUMBER', 2),
        )),
        ('PRINTC', (
            ('PRINT', 'PRINT'),
            ('ID', 'C'),
        )),
        ('C=3', (
            ('ID', 'C'),
            ('EQUALS', '='),
            ('NUMBER', 3),
        )),
        ('PRINTC', (
            ('PRINT', 'PRINT'),
            ('ID', 'C'),
        )),
        ('PRINT "A" : 10 PRINT "B"', (
            ('PRINT', 'PRINT'),
            ('STRING', 'A'),
            ('COLON', ':'),
            ('NUMBER', 10),
            ('PRINT', 'PRINT'),
            ('STRING', 'B'),
        )),
        ('B = A = 3', (
            ('ID', 'B'),
            ('EQUALS', '='),
            ('ID', 'A'),
            ('EQUALS', '='),
            ('NUMBER', 3),
        )),
        ('PRINT B', (
            ('PRINT', 'PRINT'),
            ('ID', 'B'),
        )),
        ('PRINT B; A + C', (
            ('PRINT', 'PRINT'),
            ('ID', 'B'),
            ('FLUSH', ';'),
            ('ID', 'A'),
            ('PLUS', '+'),
            ('ID', 'C'),
        )),
    ),
)
def test_lexer(lexer, test_string, expected_tokens):
    tokens = list(lexer.tokenize(test_string))

    assert len(tokens) == len(expected_tokens)

    for token, (expected_type, expected_value) in zip(tokens, expected_tokens):
        assert token.type == expected_type
        assert token.value == expected_value


@pytest.fixture
def interpreter(request):
    from basic import BasicInterpreter

    interpreter = BasicInterpreter()

    if getattr(request, 'param', None):
        lines = request.param

        if isinstance(lines, str):
            lines = [lines]

        for line in lines:
            interpreter.interpret(line)

    return interpreter


@pytest.mark.parametrize('interpreter', indirect=True, argvalues=['A = 2'])
def test_interpreter_assignment(interpreter):
    assert interpreter.variables['A'] == 2


@pytest.mark.parametrize('interpreter', indirect=True, argvalues=[(
    'A = 2',
    'A = A + 1',
)])
def test_interpreter_simple_arithmetic(interpreter):
    assert interpreter.variables['A'] == 3


@pytest.mark.parametrize('interpreter', indirect=True, argvalues=[(
    'A = 3',
    'PRINT A',
)])
# capsys goes first, otherwise parametrized interpreter output isn't captured
def test_interpreter_print_variable(capsys, interpreter):
    captured = capsys.readouterr()
    assert captured.out == '3\n'


@pytest.mark.parametrize(
    'interpreter, line_number, expected_program_line',
    indirect=['interpreter'],
    argvalues=(
        ('10 A = 5', 10, 'A = 5'),
        (' 10 A = 5', 10, 'A = 5'),
        ('10     A = 5', 10, 'A = 5'),
        ('10.2 A = 5', 10, '.2 A = 5'),
        ('20 A = 5', 20, 'A = 5'),
    )
)
def test_interpreter_add_program_line(
    interpreter,
    line_number,
    expected_program_line,
):
    assert interpreter.program[line_number] == expected_program_line


@pytest.mark.parametrize(
    'interpreter, line_number',
    indirect=['interpreter'],
    argvalues=(
        (('10 A = 5', '10'), 10),
        (('10 A = 5', '  10'), 10),
        (('10 A = 5', '10  '), 10),
    )
)
def test_interpreter_remove_program_line(interpreter, line_number):
    assert interpreter.program.get(line_number) is None


@pytest.mark.parametrize(
    'interpreter, expected_output',
    indirect=['interpreter'],
    argvalues=(
        (('10 A = 5', 'LIST'), '10 A = 5'),
        ((' 10 A = 5', 'LIST'), '10 A = 5'),
        (('10     A = 5', 'LIST'), '10 A = 5'),
        (('10.2 A = 5', 'LIST'), '10 .2 A = 5'),
        (('10 A = 1', '20 A = A + 2', 'LIST'), '10 A = 1\n20 A = A + 2'),
        (('20 A = A + 2', '10 A = 1', 'LIST'), '10 A = 1\n20 A = A + 2'),
        (('10 A = 5', 'LIST : PRINT "A"'), '10 A = 5'),
    )
)
def test_interpreter_list(capsys, interpreter, expected_output):
    captured = capsys.readouterr()
    assert captured.out == expected_output + '\n'


@pytest.mark.parametrize(
    'interpreter, expected_output',
    indirect=['interpreter'],
    argvalues=(
        ('PRINT 1 2 3', '1 2 3'),
        (('A = 3', 'PRINT A = 3'), '-1'),
        (('A = 3', 'PRINTA=3'), '-1'),
        ('PRINT"A"', 'A'),
        (('A = 3', 'PRINT"A"A'), 'A 3'),
        (('PRINT "B"'), 'B'),
        (('PRINT "B"; "B"'), 'B  B'),  # FIXME: should only have 1 space
        (('A = 2: B = 3', 'PRINT "B"; A + B'), 'B  5'),  # FIXME: should only have 1 space
    )
)
def test_print_output(capsys, interpreter, expected_output):
    captured = capsys.readouterr()
    assert captured.out == expected_output + '\n'


def test_rem_syntax_error(interpreter):
    with pytest.raises(SyntaxError):
        interpreter.interpret('PRINT"A"REM')


@pytest.mark.parametrize(
    'interpreter, expected_output',
    indirect=['interpreter'],
    argvalues=(
        ('PRINT"A":REMTHISISIGNORED', 'A'),
        ('PRINT "A" : REMTHISISIGNORED', 'A'),
        ('REM THIS IS IGNORED', ''),
        ('  REM THIS IS IGNORED', ''),
        (': REM THIS IS IGNORED', ''),
        ("' THIS IS IGNORED", ''),
    )
)
def test_rem_ignored(capsys, interpreter, expected_output):
    captured = capsys.readouterr()
    assert captured.out == expected_output + ('\n' if expected_output else '')


@pytest.mark.parametrize(
    'interpreter, expected_output',
    indirect=['interpreter'],
    argvalues=(
        (('A = 3', 'IF A THEN PRINT "B" ELSE PRINT "C"'), 'B'),
        ('IF 0 THEN PRINT "B" ELSE PRINT "C"', 'C'),
        ('IF 0 THEN PRINT "B"', ''),
        ('IF 1 THEN PRINT "B"', 'B'),
        ('IF 1 THEN PRINT "A":PRINT "B" ELSE PRINT "C":PRINT "D"', 'A\nB\nD'),
        ('IF 1 THEN PRINT "B":PRINT"C"', 'B\nC'),
        ('IF 0 THEN PRINT "B":PRINT"C"', ''),
    )
)
def test_conditionals(capsys, interpreter, expected_output):
    captured = capsys.readouterr()
    assert captured.out == expected_output + ('\n' if expected_output else '')


@pytest.mark.parametrize(
    'interpreter, expected_output',
    indirect=['interpreter'],
    argvalues=(
        ('PRINT 3 + 5 * -2', '-7'),
        ('PRINT 3 - 4 * 5', '-17'),
        ('PRINT 3 - 10 / 5', '1'),
        # ('PRINT 0.1 + 0.2', '0.3'),  # TODO: re-add this after
        #   changing to bit-for-bit parity with BASIC (using numpy floats)
        # region cb7a5a4 (Add support for ( and ) in expressions)
        ('PRINT (0.1 + 0.2) * 2', '0.6000000000000001'),  # 0.6000000000000001 float, 0.6 decimal
        ('PRINT (0.1 * 0.2) + 2', '2.02'),
        ('PRINT ((0.1 * 0.2) + (0.24 * 2))', '0.5'), # .5 float, .50 decimal
        # endregion cb7a5a4 (Add support for ( and ) in expressions)
        (('PRINT 1 - 4 * 2 ^ 3'), '-31'),  # = 1 - (4 * 8) = 1 - 32
        (('PRINT (1 - 4 * 2) ^ 3'), '-343'),  # = -7 ^ 3 = -343
    )
)
def test_extra_arithmetic(capsys, interpreter, expected_output):
    captured = capsys.readouterr()
    assert captured.out == expected_output + '\n'


@pytest.mark.parametrize(
    'interpreter, expected_output',
    indirect=['interpreter'],
    argvalues=(
        (('PRINTC', 'C=3', 'PRINTC'), '0\n3'),
    )
)
def test_default_variable_value(capsys, interpreter, expected_output):
    captured = capsys.readouterr()
    assert captured.out == expected_output + '\n'


def test_program_line_mid_statement_invalid(interpreter):
    with pytest.raises(SyntaxError):
        interpreter.interpret('PRINT "A" : 10 PRINT "B"')


@pytest.mark.parametrize(
    'interpreter, expected_output',
    indirect=['interpreter'],
    argvalues=(
        (('A = 3', 'B = A = 3', 'PRINT B'), '-1'),
        (('A = 2', 'B = A = 3', 'PRINT B'), '0'),
    )
)
def test_boolean_values(capsys, interpreter, expected_output):
    captured = capsys.readouterr()
    assert captured.out == expected_output + '\n'


@pytest.mark.parametrize(
    'interpreter, expected_output',
    indirect=['interpreter'],
    argvalues=[('PRINT "A" : PRINT "B"', 'A\nB')],
)
def test_multiple_statements(capsys, interpreter, expected_output):
    captured = capsys.readouterr()
    assert captured.out == expected_output + '\n'


@pytest.mark.parametrize('interpreter', indirect=True, argvalues=[''])
def test_empty_statement(capsys, interpreter):
    captured = capsys.readouterr()
    assert captured.out == ''


def test_identifier_only_statement_invalid(interpreter):
    with pytest.raises(SyntaxError):
        interpreter.interpret('ABCDEF')


@pytest.mark.parametrize(
    'interpreter, expected_output',
    indirect=['interpreter'],
    argvalues=(
        ('RUN', ''),
        (('10 PRINT "A"', 'RUN'), 'A'),
        (('10 PRINT "A"', '20 PRINT "B"', 'RUN'), 'A\nB'),
    )
)
def test_interpreter_run(capsys, interpreter, expected_output):
    captured = capsys.readouterr()
    assert captured.out == expected_output + ('\n' if expected_output else '')


@pytest.mark.parametrize(
    'interpreter, expected_output',
    indirect=['interpreter'],
    argvalues=(
        (('10 PRINT "A"', 'GOTO 10'), 'A'),
        (('10 PRINT "A"', '20 PRINT "B"', '30 PRINT "C"', 'GOTO 20'), 'B\nC'),
        (('10 GOTO 20', '20 PRINT "B"', '30 PRINT "C"', 'GOTO 10'), 'B\nC'),
    )
)
def test_interpreter_goto(capsys, interpreter, expected_output):
    captured = capsys.readouterr()
    assert captured.out == expected_output + '\n'


def test_goto_invalid_argument(interpreter):
    with pytest.raises(SyntaxError):
        interpreter.interpret('GOTO "ABC"')
