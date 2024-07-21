import collections
import math
import numpy as np
import sys

import sly
from sly.lex import Lexer, Token
from sly.yacc import Parser

Variable = collections.namedtuple('Variable', ['name'])
Expression = collections.namedtuple('Expression', ['operation', 'arguments'])
Statement = collections.namedtuple('Statement', ['operation', 'arguments'])
ControlCharacter = collections.namedtuple('Statement', ['operation', 'arguments'])


class BasicLexer(Lexer):
    tokens = {
        ID,
        REM,
        PRINT,
        FLUSH,
        DIM,
        REDIM,
        AS,
        TYPE_NAME,  # A specific type, *not* the TYPE statement
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
        RPAREN,
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

    AS = r'[Aa][Ss]'
    TYPE_NAME = r'([Ii][Nn][Tt][Ee][Gg][Ee][Rr])|([Ll][Oo][Nn][Gg])|([Ss][Ii][Nn][Gg][Ll][Ee])|([Dd][Oo][Uu][Bb][Ll][Ee])|([Ss][Tt][Rr][Ii][Nn][Gg])'
    REM = r"(?:[Rr][Ee][Mm]|').*"
    DIM = r'[Dd][Ii][Mm]'
    REDIM = r'[Rr][Ee][Dd][Ii][Mm]'
    PRINT = r'[Pp][Rr][Ii][Nn][Tt]'
    IF = r'[Ii][Ff]'
    THEN = r'[Tt][Hh][Ee][Nn]'
    ELSE = r'[Ee][Ll][Ss][Ee]'
    LIST = r'[Ll][Ii][Ss][Tt]'
    RUN = r'[Rr][Uu][Nn]'
    GOTO = r'[Gg][Oo][Tt][Oo]'

    # ID = r'[A-Za-z_][A-Za-z0-9_]*'  # does not include type suffixes
    ID = r'[a-zA-Z_][a-zA-Z0-9_]*[$%&!#]*'  # includes type suffixes

    @_(r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)')
    def NUMBER(self, token):
        if(
            self.index
            and self.text[:token.index].strip() != ""
        ):
            # If it is *not* a line number
            float_value = np.float64(token.value)
            int_value = np.int32(float_value)
            token.value = (
                int_value
                if (np.float64(int_value) == float_value)  # math.isclose(int_value, float_value)
                else float_value
            )
        else:  # LINENO
            for part in token.value.split("."):
                if not part.isdigit():
                    raise SyntaxError(
                        "{} special characters are not allowed in LINENO {}"
                        .format(part, token.value)
                    )
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
    FRIENDLY_TYPES = {
        'ID': "identifier",
    }
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

    @_('DIM ID AS TYPE_NAME MULTIPLY expr')
    def statement(self, parsed):
        return Statement('dim', ['DIM', parsed.ID, parsed.TYPE_NAME, parsed.expr])

    @_('DIM ID AS TYPE_NAME')
    def statement(self, parsed):
        # TODO: allow String * NUMBER to set length
        return Statement('dim', ['DIM', parsed.ID, parsed.TYPE_NAME])

    @_('DIM ID')
    def statement(self, parsed):
        # TODO: allow String * NUMBER to set length
        return Statement('dim', ['DIM', parsed.ID])

    @_('REDIM ID AS TYPE_NAME MULTIPLY expr')
    def statement(self, parsed):
        return Statement('dim', ['REDIM', parsed.ID, parsed.TYPE_NAME, parsed.expr])

    @_('REDIM ID AS TYPE_NAME')
    def statement(self, parsed):
        # TODO: allow String * NUMBER to set length
        return Statement('dim', ['REDIM', parsed.ID, parsed.TYPE_NAME])

    @_('REDIM ID')
    def statement(self, parsed):
        # TODO: allow String * NUMBER to set length
        return Statement('dim', ['REDIM', parsed.ID])

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
        """Causes flushing previous PRINT args like BASIC.

        Returns:
            ControlCharacter: check for that type in def print in parser
        """
        return ControlCharacter('flush', [])

    @_('variable')
    def expr(self, parsed):
        return Expression('get_variable', [parsed.variable.name])

    @_('ID')
    def variable(self, parsed):
        return Variable(parsed.ID)

    def error(self, token):
        if not token:
            raise EOFError('Parse error in input, unexpected EOF')
        friendly_type = BasicParser.FRIENDLY_TYPES.get(token.type)
        if not friendly_type:
            if isinstance(token.type, str):
                # TODO: Why does this happen? Should .type = 'LINENO' be changed?
                friendly_type = token.type
            else:
                friendly_type = token.type.__name__
        if (isinstance(token.value, str)
                and (friendly_type.upper() == token.value.upper())):
            # It is a statement and the statement is the same as the token.
            # (in another case the value could be a number etc)
            raise SyntaxError(
                'Unexpected "{}"'
                .format(token.value)
            )
        else:
            raise SyntaxError(
                'Unexpected {} "{}"'
                .format(friendly_type, token.value)
            )
        # token.lineno


class BasicInterpreter:
    """The real-time state of the BASIC program.

    Attributes:
        typed_names (dict[str]): Key is variable name without suffix,
            and value is name with suffix. If suffix is changed
            (which is allowed in BASIC), the old variable name in the
            value should be deleted from self.variables.
    """
    DEFAULT_TYPE = np.float32  # QB Single
    SUFFIX_TYPES = {
        '%': np.int16,  # QB Integer
        '&': np.int32,  # QB Long
        '!': np.float32,  # QB Single
        '#': np.float64,  # QB Double
        '$': str,  # QB String
    }
    TYPE_NAME_TYPES = {
        'INTEGER': np.int16,
        'LONG': np.int32,
        'SINGLE': np.float32,
        'DOUBLE': np.float64,
        'STRING': str,
    }
    def __init__(self):
        self.lexer = BasicLexer()
        self.parser = BasicParser(self)
        self.variables = collections.defaultdict(int)
        self.program = {}
        self.running_program = False
        self.typed_names = {}
        self.strlen = {}
        self.types = {}

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

        # if isinstance(return_value, (float, np.float32, np.float64)):
        #     int_return_value = int(return_value)
        #     return_value = (
        #         int_return_value
        #         if math.isclose(int_return_value, return_value)
        #         else return_value
        #     )

        return return_value

    def negative(self, a):
        return -a

    def add(self, a, b):
        if not isinstance(a, type(b)):
            # TODO: test type coercion & make sure it matches BASIC
            #   (Should be fine since typically this would happen in a PRINT statement)
            if isinstance(a, str):
                if not isinstance(b, str):
                    print("Warning: Coercing {} to {}"
                        .format(type(b).__name__, type(a).__name__),
                        file=sys.stderr)
                    b = type(a)(b)
            elif isinstance(b, str):
                if not isinstance(a, str):
                    print("Warning: Coercing {} to {}"
                        .format(type(a).__name__, type(b).__name__),
                        file=sys.stderr)
                    a = type(b)(a)
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

    def flush(self):
        # TODO: low-pri: Actual BASIC would previous PRINT args first
        sys.stdout.flush()

    def get_variable(self, name):
        return self.variables.get(name, 0)

    @staticmethod
    def split_type(name):
        """Split ID or NUMBER from type specifier suffix"""
        last_backquote = name.find("`")
        if last_backquote > -1:
            # multi-bit type has ` followed by #of bits.
            raise NotImplementedError("MULTIBIT type is not implemented.")
            return name[:last_backquote], name[last_backquote:]
        if "~" in name:
            raise NotImplementedError("unsigned types are not implemented.")
        bare_name = name.rstrip("`~%#!&$")
        if bare_name != name:
            return bare_name, name[len(bare_name)-len(name):]  # negative slice from diff
        else:
            # otherwise the diff isn't negative, so don't slice improperly
            return name, ""

    def set_variable(self, name, value):
        # print("[set_variable] {}={}({})"
        #       .format(name, type(value).__name__, value))
        bare_name, suffix = BasicInterpreter.split_type(name) # name[-1:]
        if suffix:
            suffix_type = BasicInterpreter.SUFFIX_TYPES.get(suffix)
            if not suffix_type:
                raise NotImplementedError("Type suffix {} is not implemented."
                                          .format(suffix))
            # if name[-2:-1] in BasicInterpreter.SUFFIX_TYPES:
            #     raise NotImplementedError(
            #         "{} suffix is not implemented"
            #         .format(name[2:]))
            #     # TODO: implement more suffixes:
            #     # <https://qb64.dijkens.com/wiki/Data_types.html#Data_type_limits>
            # ^ More than 1 (allowed in BASIC) is already blocked by the
            #   ID regex in Lexer
            old_name = self.typed_names.get(bare_name)
            if old_name is not None:
                if old_name != name:
                    # NOTE: QB allows type redefinition
                    print("Warning: Overriding {} type with redeclaration {}"
                          .format(old_name, name))
                    del self.variables[old_name]
                    self.typed_names[bare_name] = name
            else:
                self.typed_names[bare_name] = name
            this_type = suffix_type
            # ^ causes "DeprecationWarning: NumPy will stop allowing conversion of out-of-bound Python integers to integer arrays.  The conversion of 65536 to int16 will fail in the future.
            #   For the old behavior, usually:
            #       np.array(value).astype(dtype)`
            #   will give the desired result (the cast overflows).
            #     self.variables[name] = suffix_type(self.evaluate(value))

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
            this_type = self.types.get(bare_name)
            # TODO: lookup type of first letter of bare_name as set by `DEF* criteria`)
            if this_type is None:
                this_type = BasicInterpreter.DEFAULT_TYPE
                # print("Warning: Used default type {} for {}"
                #       .format(this_type.__name__, bare_name))

        final_value = self.evaluate(value)
        if this_type is str:
            if not isinstance(final_value, str):
                raise TypeError(
                    "Illegal string-number conversion {}({}, {}) = {}({})"
                    .format(this_type, bare_name, suffix,
                            type(final_value).__name__, final_value))
        elif isinstance(final_value, str):
            if this_type is not str:
                raise TypeError(
                    "Illegal string-number conversion {}({}, {}) = {}({})"
                    .format(this_type.__name__, bare_name, suffix,
                            type(final_value).__name__, final_value))

        # print("[set_variable] {}={}({})"
        #       .format(name, type(final_value).__name__, final_value),
        # )  # file=sys.stderr)

        self.variables[name] = this_type(final_value)

    def compare_variable(self, name, value):
        # print("[compare_variable] {} == {}\n"
        #       .format(self.variables[name], value),
        # )#file=sys.stderr)
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

    def dim(self, *args):
        keyword = args[0]
        id = args[1]
        type_name = args[2] if len(args) > 2 else None
        length = args[3] if len(args) > 3 else None
        this_type = None
        if len(args):
            if not type_name:
                raise SyntaxError("missing type specifier after AS")
            this_type = BasicInterpreter.TYPE_NAME_TYPES.get(type_name.upper())
        if this_type is None:
            raise SyntaxError(
                "invalid type \"{}\" specified".format(this_type))
        bare_name, suffix = BasicInterpreter.split_type(id)
        if suffix:
            if (len(args) > 2):
                raise SyntaxError(
                    "Using a type suffix ('{}') for \"{}\" with `AS` is redundant."
                    .format(suffix, id))
            this_type = BasicInterpreter.SUFFIX_TYPES.get(suffix)
            if this_type is None:
                raise NotImplementedError("Type suffix {} is invalid."
                                          .format(suffix))
        elif this_type is None:
            this_type = BasicInterpreter.DEFAULT_TYPE
        old_name = self.typed_names.get(bare_name)
        if keyword == "DIM":
            if old_name is not None:
                raise SyntaxError(
                    "redeclaration of {} (declared as {}) requires REDIM"
                    .format(id, old_name)
                )
            elif id in self.variables:
                raise SyntaxError(
                    "redeclaration of {} requires REDIM"
                    .format(id)
                )
            if id in self.strlen:
                raise NotImplementedError(
                    "cleanup failed for {} length={}"
                    .format(id, self.strlen[id])
                )
        elif keyword == "REDIM":
            if old_name is not None:
                # TODO: Mimic BASIC behavior here (behavior is unknown).
                print("Warning: {} is overwriting {} with {}"
                      .format(keyword, old_name, id),
                      file=sys.stderr)
                del self.typed_names[bare_name]
                del self.variables[old_name]
            if id in self.strlen:
                del self.strlen[id]
        else:
            raise NotImplementedError("Expected DIM or REDIM but got {}"
                                      .format(keyword))
        if type_name.upper() == "STRING":
            self.variables[id] = this_type("")
            if length:
                self.strlen[id] = length
        else:
            self.variables[id] = this_type(0)
            if length:
                raise SyntaxError("`* length` is only valid for STRING.")
        self.types[bare_name] = this_type  # type_name.upper()

    def print(self, *args):
        """Print with a space around numbers like BASIC.
        There are no spaces around strings in BASIC.
        """
        # TODO: Make number formatting configurable (Only if BASIC has a way)
        length = 0
        for arg in args:
            if isinstance(arg, ControlCharacter):
                if arg.operation == 'flush':
                    sys.stdout.flush()
                else:
                    raise NotImplementedError(
                        "Unknown ControlCharacter in print: {}({})"
                        .format(type(arg).__name__, self.evaluate(arg))
                    )
            else:
                value = self.evaluate(arg)
                if "{}".format(value) == "":
                    print("Warning: got '' from {}".format(arg),
                          file=sys.stderr)
                if isinstance(value, (np.float32, np.float64)):
                    v_str = "{:7f}".format(value).rstrip("0.")
                    if not v_str:
                        v_str = " 0 "
                        length = 3
                        sys.stdout.write(v_str)
                    elif value < 0:
                        length += len("{} ".format(v_str).strip())
                        sys.stdout.write("{} ".format(v_str))
                    else:
                        length += len(" {} ".format(v_str).strip())
                        sys.stdout.write(" {} ".format(v_str))
                    assert length > 0
                elif isinstance(value, (np.int16, np.int32, np.int64)):
                    if isinstance(value, np.int64):
                        print("Warning: {} was automatically upcast"
                              " to numpy.int64({})"
                              .format(arg, value), file=sys.stderr)
                        # TODO: Figure out why this happens. It is only
                        #   known to happen by adding a literal (or
                        #   Python int) to a numpy.int32.
                    if "{}".format(value) == "":
                        raise NotImplementedError(
                            "{} reverted to blank '{}'"
                            .format(arg, value))
                    if value < 0:
                        length += len("{} ".format(value).strip())
                        sys.stdout.write("{} ".format(value))
                    else:
                        length += len(" {} ".format(value).strip())
                        sys.stdout.write(" {} ".format(value))
                    assert length > 0
                elif isinstance(value, int):
                    # raise NotImplementedError(
                    #     "Numpy was skipped (got {}({}))"
                    #     .format(type(value).__name__, value))
                    # Returns from comparison
                    # Example: "B = 3: PRINT B = 3" (yields "-1 ")
                    if "{}".format(value) == "":
                        raise NotImplementedError(
                            "{} reverted to blank '{}'"
                            .format(arg, value))
                    if value < 0:
                        length += len("{} ".format(value).strip())
                        sys.stdout.write("{} ".format(value))
                    else:
                        length += len(" {} ".format(value).strip())
                        sys.stdout.write(" {} ".format(value))
                    assert length > 0
                elif isinstance(value, float):
                    raise NotImplementedError(
                        "Numpy was skipped (got {}({}))"
                        .format(type(value).__name__, value))
                elif isinstance(value, str):
                    length += len(value)
                    sys.stdout.write(value)
                else:
                    # ok probably
                    raise NotImplementedError(
                        "Unknown type {}({})"
                        .format(type(value).__name__, value))
                    length += len(value)
                    sys.stdout.write(value)
        # if len(args) == 0:
        #     raise NotImplementedError("No args in PRINT")
        # if length == 0:
        #     # ok probably
        #     raise NotImplementedError(
        #         "args {} became ''".format(args))
        print("")

    def list(self):
        for lineno, line in sorted(self.program.items()):
            print(f'{lineno} {line}')

if __name__ == "__main__":
    raise RuntimeError(
        "The {} module should be imported not run."
        " Use main.py to run a file."
        .format(__file__))