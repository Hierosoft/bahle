import collections
import math
import numpy as np
import sys

from sly.lex import Lexer, Token
from sly.yacc import Parser


Variable = collections.namedtuple('Variable', ['name'])
Expression = collections.namedtuple('Expression', ['operation', 'arguments'])
Statement = collections.namedtuple('Statement', ['operation', 'arguments'])


class BasicLexer(Lexer):
    tokens = {
        ID,
        REM,
        PRINT,
        FLUSH,
        IF,
        THEN,
        ELSE,
        LIST,
        RUN,
        GOTO,
        STRING,
        LINENO,
        NUMBER,
        PLUS,
        MINUS,
        MULTIPLY,
        DIVIDE,
        POWER,
        EQUALS,
        COLON,
        LPAREN,
        RPAREN
    }

    ignore = ' '

    FLUSH = r";"
    PLUS = r'\+'
    MINUS = r'-'
    MULTIPLY = r'\*'
    DIVIDE = r'/'
    POWER = r'\^'
    EQUALS = r'='
    COLON = r':'
    LPAREN = r'\('
    RPAREN = r'\)'

    REM = r"(?:REM|').*"
    PRINT = r'PRINT'
    IF = r'IF'
    THEN = r'THEN'
    ELSE = r'ELSE'
    LIST = r'LIST'
    RUN = r'RUN'
    GOTO = r'GOTO'

    # ID = r'[A-Za-z_][A-Za-z0-9_]*'  # does not include type suffixes
    ID = r'[a-zA-Z_][a-zA-Z0-9_]*[$%&!#]*'  # includes type suffixes

    @_(r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)')
    def NUMBER(self, token):
        if(
            self.index
            and self.text[:token.index] != token.index * ' '
        ):
            float_value = float(token.value)
            int_value = int(float_value)
            token.value = (
                int_value
                if math.isclose(int_value, float_value)
                else float_value
            )

        else:
            if '.' not in token.value:
                token.value = int(token.value)

            else:
                dot_index = token.value.index('.')
                self.index -= len(token.value) - dot_index
                token.value = int(token.value[:dot_index])

            token.type = 'LINENO'

            if self.text[self.index:].strip(' '):
                self.begin(LineLexer)

        return token

    @_(r'"[^"]*"?')
    def STRING(self, token):
        token.value = token.value[1:]

        if token.value.endswith('"'):
            token.value = token.value[:-1]
        else:
            raise SyntaxError("Unterminated string (missing '\"')")

        return token


class LineLexer(Lexer):
    tokens = {LINE}
    ignore = ' '

    @_(r'.+')
    def LINE(self, token):
        self.begin(BasicLexer)
        return token


class BasicParser(Parser):
    tokens = BasicLexer.tokens.union(LineLexer.tokens)
    precedence = (
        ('nonassoc', IF, THEN),
        ('left', COLON),
        ('nonassoc', ELSE),
        ('left', EQUALS),
        ('nonassoc', FLUSH),
        ('left', CREATE_EXPRS, APPEND_EXPRS),
        ('left', PLUS, MINUS),
        ('left', MULTIPLY, DIVIDE),
        ('left', POWER),  # exponent (*not* same as EXP function)
        ('nonassoc', UNARY_MINUS),
    )

    def __init__(self, interpreter):
        self.interpreter = interpreter

    @_('statement')
    def statements(self, parsed):
        if parsed.statement:
            return [parsed.statement]

    @_('statements COLON statement')
    def statements(self, parsed):
        parsed.statements.append(parsed.statement)
        return parsed.statements

    @_(
        'statements COLON empty',
        'empty COLON statements',
    )
    def statements(self, parsed):
        return parsed.statements

    @_('')
    def empty(self, parsed):
        pass

    @_('LINENO LINE')
    def statement(self, parsed):
        return Statement('add_program_line', (parsed.LINENO, parsed.LINE))

    @_('LINENO')
    def statement(self, parsed):
        return Statement('remove_program_line', [parsed.LINENO])

    @_('IF expr THEN statements')
    def statement(self, parsed):
        return Statement('conditional', (parsed.expr, parsed.statements))

    @_('IF expr THEN statements ELSE statement')
    def statement(self, parsed):
        return Statement(
            'conditional',
            (parsed.expr, parsed.statements, parsed.statement),
        )

    @_('variable EQUALS expr')
    def statement(self, parsed):
        return Statement('set_variable', (parsed.variable.name, parsed.expr))

    @_('REM')
    def statement(self, parsed):
        return Statement('noop', [])

    @_('PRINT exprs')
    def statement(self, parsed):
        return Statement('print', parsed.exprs)

    @_('LIST')
    def statement(self, parsed):
        return Statement('list', [])

    @_('RUN')
    def statement(self, parsed):
        return Statement('run_program', [])

    @_('GOTO expr')
    def statement(self, parsed):
        return Statement('goto', [parsed.expr])

    @_('expr %prec CREATE_EXPRS')
    def exprs(self, parsed):
        return [parsed.expr]

    @_('exprs expr %prec APPEND_EXPRS')
    def exprs(self, parsed):
        parsed.exprs.append(parsed.expr)
        return parsed.exprs

    @_('variable EQUALS expr')
    def expr(self, parsed):
        return Expression(
            'compare_variable',
            [parsed.variable.name, parsed.expr],
        )

    @_('MINUS expr %prec UNARY_MINUS')
    def expr(self, parsed):
        return Expression('negative', [parsed.expr])

    @_('LPAREN expr RPAREN')
    def expr(self, parsed):
        return parsed.expr

    @_('expr PLUS expr')
    def expr(self, parsed):
        return Expression('add', [parsed.expr0, parsed.expr1])

    @_('expr MINUS expr')
    def expr(self, parsed):
        return Expression('subtract', [parsed.expr0, parsed.expr1])

    @_('expr MULTIPLY expr')
    def expr(self, parsed):
        return Expression('multiply', [parsed.expr0, parsed.expr1])

    @_('expr DIVIDE expr')
    def expr(self, parsed):
        return Expression('divide', [parsed.expr0, parsed.expr1])

    @_('expr POWER expr')
    def expr(self, parsed):
        return Expression('power', [parsed.expr0, parsed.expr1])

    @_(
        'NUMBER',
        'STRING',
    )
    def expr(self, parsed):
        return parsed[0]

    @_('FLUSH')
    def expr(self, parsed):
        # return Statement('flush', [])
        # TODO: low-priority: Actual BASIC would previous PRINT args first
        #   but returning a Statement here causes it to show.
        #   - Add a new class and return an instance?
        #   - Either way it would require alternating between
        #     sys.stdout.flush() and sys.stdout.write(...) then a
        #     newline and an auto flush at the end of the line.
        return ""

    @_('variable')
    def expr(self, parsed):
        return Expression('get_variable', [parsed.variable.name])

    @_('ID')
    def variable(self, parsed):
        return Variable(parsed.ID)

    def error(self, token):
        if not token:
            raise EOFError('Parse error in input, unexpected EOF')

        raise SyntaxError(
            f'Syntax error at line {token.lineno}, token={token.type}'
        )


class BasicInterpreter:
    """The real-time state of the BASIC program.

    Attributes:
        typed_names (dict[str]): Key is variable name without suffix,
            and value is name with suffix. If suffix is changed
            (which is allowed in BASIC), the old variable name in the
            value should be deleted from self.variables.
    """
    SUFFIX_TYPES = {
        '%': np.int16,  # QB Integer
        '&': np.int32,  # QB Long integer
        '!': np.float32,  # QB Single
        '#': np.float64,  # QB Double
    }
    def __init__(self):
        self.lexer = BasicLexer()
        self.parser = BasicParser(self)
        self.variables = collections.defaultdict(int)
        self.program = {}
        self.running_program = False
        self.typed_names = {}

    def interpret(self, line):
        try:
            statements = self.parser.parse(self.lexer.tokenize(line))

        except EOFError:
            raise SyntaxError('Unexpected EOF')

        for statement in statements:
            self.execute(*statement)

            if statement.operation in ('list', 'run_program', 'goto'):
                break

    def execute(self, instruction, arguments):
        return getattr(self, instruction)(*arguments)

    def evaluate(self, expression):
        evaluation_stack = collections.deque()
        argument_index_stack = collections.deque()
        node = expression
        last_visited_node = None

        while evaluation_stack or node is not None:
            if node is not None:
                evaluation_stack.append(node)

                if isinstance(node, Expression):
                    argument_index_stack.append(0)
                    node = node.arguments[0]
                else:
                    node = None

            else:
                next_node = evaluation_stack[-1]

                if(
                    isinstance(next_node, Expression)
                    and len(next_node.arguments) > 1
                    and last_visited_node != next_node.arguments[1]
                ):
                    argument_index_stack.append(1)
                    node = next_node.arguments[1]

                elif argument_index_stack:
                    evaluation_stack[-1].arguments[
                        argument_index_stack.pop()
                    ] = last_visited_node = self.visit(evaluation_stack.pop())

                else:
                    return self.visit(next_node)

    def visit(self, node):
        return_value = node

        if isinstance(node, Expression):
            return_value = self.execute(*node)

        if isinstance(return_value, float):
            int_return_value = int(return_value)
            return_value = (
                int_return_value
                if math.isclose(int_return_value, return_value)
                else return_value
            )

        return return_value

    def negative(self, a):
        return -a

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        return a / b

    def power(self, a, b):
        # NOTE: In Python, `^` is bitwise XOR.
        # Also, pow differs from math.pow and **
        #   (See <https://stackoverflow.com/a/10282852/4541104>):
        # math.pow(0.000000,-2.200000)
        #     ValueError: math domain error
        # __builtins__.pow(0.000000,-2.200000)
        #     ZeroDivisionError: 0.0 cannot be raised to a negative power
        # ** uses the object's own method, such as if
        #    __pow__(), __rpow__() or __ipow__() are overridden.
        return a ** b

    # def flush(self):
    #     # TODO: low-pri: Actual BASIC would previous PRINT args first
    #     sys.stdout.flush()

    def get_variable(self, name):
        return self.variables.get(name, 0)

    def set_variable(self, name, value):
        suffix = name[-1:]
        suffix_type = BasicInterpreter.SUFFIX_TYPES.get(suffix)
        # TODO: lookup type of variable (as set by `DEF*`` and `as *`)
        if suffix_type:
            # if name[-2:-1] in BasicInterpreter.CAST:
            #     raise NotImplementedError(
            #         "{} suffix is not implemented"
            #         .format(name[2:]))
            #     # TODO: implement more suffixes:
            #     # <https://qb64.dijkens.com/wiki/Data_types.html#Data_type_limits>
            # ^ More than 1 (allowed in BASIC) is already blocked by the
            #   ID regex in Lexer
            part0 = name.rstrip(name[-1:])
            old_name = self.typed_names.get(part0)
            if old_name is not None:
                if old_name != name:
                    # NOTE: QB allows type redefinition
                    print("Warning: Overriding {} type with redeclaration {}"
                          .format(old_name, name))
                    del self.variables[old_name]
                    self.typed_names[part0] = name
            else:
                self.typed_names[part0] = name
            self.variables[name] = suffix_type(self.evaluate(value))
            # ^ causes "DeprecationWarning: NumPy will stop allowing conversion of out-of-bound Python integers to integer arrays.  The conversion of 65536 to int16 will fail in the future.
            #   For the old behavior, usually:
            #       np.array(value).astype(dtype)`
            #   will give the desired result (the cast overflows).
            #     self.variables[name] = cast(self.evaluate(value))

            # -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html"
            # Therefore:
            # self.variables[name] = np.array(self.evaluate(value)).astype(suffix_type)
            # FIXME: still causes:
            # DeprecationWarning: NumPy will stop allowing conversion of out-of-bound Python integers to integer arrays.  The conversion of 65536 to int16 will fail in the future.
            #   For the old behavior, usually:
            #       np.array(value).astype(dtype)`
            #   will give the desired result (the cast overflows).
            #     self.variables[name] = suffix_type(self.evaluate(value))
            # Maybe just require numpy <= 1.21.5 ...
        else:
            self.variables[name] = self.evaluate(value)

    def compare_variable(self, name, value):
        return -1 if self.variables[name] == value else 0

    def add_program_line(self, lineno, line):
        self.program[lineno] = line

    def remove_program_line(self, lineno):
        self.program.pop(lineno, None)

    def run_program(self, lineno=None):
        if not self.program:
            return

        self.running_program = True

        linenos = sorted(self.program)
        current_line_index = 0
        self.current_program_lineno = linenos[0]

        if lineno is not None:
            current_line_index = linenos.index(lineno)
            self.current_program_lineno = lineno

        while True:
            if self.current_program_lineno is not None:
                current_line_index = linenos.index(self.current_program_lineno)

            else:
                try:
                    current_line_index += 1
                    self.current_program_lineno = linenos[current_line_index]

                except IndexError:
                    break

            current_program_line = self.program[self.current_program_lineno]
            self.last_program_lineno = self.current_program_lineno
            self.current_program_lineno = None
            self.interpret(current_program_line)

        self.running_program = False

    def goto(self, expr):
        try:
            int(expr)

        except ValueError:
            raise SyntaxError('Type mismatch error')

        if not self.running_program:
            self.run_program(lineno=int(expr))

        else:
            self.current_program_lineno = int(expr)

    def conditional(self, expr, then_statements, else_statement=None):
        if self.evaluate(expr):
            for statement in then_statements:
                self.execute(*statement)

        elif else_statement:
            self.execute(*else_statement)

    def noop(self):
        pass

    def print(self, *args):
        print(*(self.evaluate(arg) for arg in args))

    def list(self):
        for lineno, line in sorted(self.program.items()):
            print(f'{lineno} {line}')
