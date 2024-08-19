# -*- coding: utf-8 -*-
# ruff: noqa: F811, F821
# F811 is ignored since "statement" is defined many times in sly
# F821 is reduced to information since constants and _ decorator are
#   auto-defined sly, but not set to False in case there are real issues.
# - Setting `_ = None` prevents highlighting every use of the decorator.
# - same for Pylance reportUndefinedVariable
# - A line-by-line solution may need `# pylint: disable=undefined-variable`
# pyright: reportUndefinedVariable=information

import collections

from sly.yacc import Parser

from bahle.basiclexer import (
    BasicLexer,
    LineLexer,
)

_ = None

Variable = collections.namedtuple('Variable', ['name'])
Expression = collections.namedtuple('Expression', ['operation', 'arguments'])
Statement = collections.namedtuple('Statement', ['operation', 'arguments'])
ControlCharacter = collections.namedtuple('Statement', ['operation', 'arguments'])


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
