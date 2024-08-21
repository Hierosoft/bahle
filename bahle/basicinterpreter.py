# -*- coding: utf-8 -*-
import collections
# import math
import numpy as np
import sys

from logging import getLogger

from bahle.basiclexer import BasicLexer
from bahle.basicparser import (
    BasicParser,
    ControlCharacter,
    Expression,
    Statement,
    Variable,
)

logger = getLogger(__name__)

alphabet_lower = "abcdefghijklmnopqrstuvwxyz"
alphabet_upper = alphabet_lower.upper()
alphabet_case_insensitive = alphabet_lower + alphabet_upper


def get_names(*args):
    names = []
    expr_i = 0
    statement_i = 0
    other_i = 0
    for arg in args:
        if isinstance(arg, Variable):
            names.append(arg.name)
        elif isinstance(arg, Expression):
            names.append("expr{}".format(expr_i))
            expr_i += 1
        elif isinstance(arg, Statement):
            names.append("statement{}".format(statement_i))
            statement_i += 1
        else:
            names.append("other{}.{}".format(other_i, type(arg).__name__))
            other_i += 1
    return names


def assert_compare_allowed(*args, names=None):
    allowed_types = tuple(BasicInterpreter.TYPE_NAME_TYPES.values())
    if not names:
        if len(args) == 2:
            names = ('a', 'b')
        elif len(args) <= 26:
            names = alphabet_lower
        else:
            names = ["" for i in range(len(args))]
            # ^ blank (index is below)
    for i, var in enumerate(args):
        if not isinstance(var, allowed_types):
            raise TypeError(
                "error:_ operand[{}]{} is {}({})"
                .format(i, names[i], type(var).__name__, var))


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

    def to_value(self, expression):
        # TODO: I don't know if this should be in evaluate, but was
        #   necessary for comparison functions -Poikilos
        node = self.evaluate(expression)
        if isinstance(node, Variable):
            return self.variables[node.name]
        return node

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

    # def variable_equals_value(self, name, value):
    #     # print("[any_equals_any] {} == {}\n"
    #     #       .format(self.variables[name], value),
    #     # )#file=sys.stderr)
    #     return -1 if self.variables[name] == value else 0

    def any_equals_any(self, a, b):
        # print("[any_equals_any] {} == {}\n"
        #       .format(self.variables[name], value),
        # )#file=sys.stderr)
        names = get_names(a, b)
        a = self.to_value(a)
        b = self.to_value(b)
        assert_compare_allowed(a, b, names=names)
        # return -1 if self.evaluate(a) == self.evaluate(b) else 0
        return -1 if a == b else 0

    def any_gt_any(self, a, b):
        # logger.info(
        #     "PREVIEW: {}({}) GT {}({})"
        #     .format(type(a).__name__, a,
        #             type(b).__name__, b))
        names = get_names(a, b)
        a = self.to_value(a)
        b = self.to_value(b)
        assert_compare_allowed(a, b, names=names)
        # return -1 if self.evaluate(a) > self.evaluate(b) else 0
        return -1 if a > b else 0

    def any_lt_any(self, a, b):
        # logger.info(
        #     "PREVIEW: {}({}) LT {}({})"
        #     .format(type(a).__name__, a,
        #             type(b).__name__, b))
        names = get_names(a, b)
        a = self.to_value(a)
        b = self.to_value(b)
        assert_compare_allowed(a, b, names=names)
        # return -1 if self.evaluate(a) < self.evaluate(b) else 0
        return -1 if a < b else 0

    def any_ge_any(self, a, b):
        names = get_names(a, b)
        a = self.to_value(a)
        b = self.to_value(b)
        assert_compare_allowed(a, b, names=names)
        # return -1 if self.evaluate(a) >= self.evaluate(b) else 0
        return -1 if a >= b else 0

    def any_le_any(self, a, b):
        names = get_names(a, b)
        a = self.to_value(a)
        b = self.to_value(b)
        assert_compare_allowed(a, b, names=names)
        # return -1 if self.evaluate(a) <= self.evaluate(b) else 0
        return -1 if a <= b else 0

    def any_ne_any(self, a, b):
        names = get_names(a, b)
        a = self.to_value(a)
        b = self.to_value(b)
        assert_compare_allowed(a, b, names=names)
        # return -1 if self.evaluate(a) != self.evaluate(b) else 0
        return -1 if a != b else 0

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
