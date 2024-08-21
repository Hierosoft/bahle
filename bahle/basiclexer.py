# -*- coding: utf-8 -*-
# ruff: noqa: F811, F821
# F811 is ignored since "statement" is defined many times in sly
# F821 is reduced to information since constants and _ decorator are
#   auto-defined sly, but not set to False in case there are real issues.
# - Setting `_ = None` prevents highlighting every use of the decorator.
# - same for Pylance reportUndefinedVariable
# - A line-by-line solution may need `# pylint: disable=undefined-variable`
# pyright: reportUndefinedVariable=information
import numpy as np

from sly.lex import Lexer

_ = None


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
        GT,
        LT,
        GE,
        LE,
        NE,
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
    # Below are order-dependent (patterns starting with another!):
    NE = r'<>'
    LE = r'<='
    GE = r'>='
    GT = r'>'
    LT = r'<'
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
