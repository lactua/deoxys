from _ast import Call, ClassDef
import types
import ctypes
from random import choices, randint, shuffle, choice
from string import ascii_letters, digits
import ast
from typing import Any
from zlib import compress
from os import listdir, remove, mkdir
from python_minifier import minify
from tokenize import tokenize
from os.path import isdir
from time import time_ns

class StepFollower:
    def __init__(self, dir, ext='txt') -> None:
        self.counter = 0
        self.dir = dir
        self.ext = ext
        
        if not isdir(dir):
            mkdir(dir)

        for file in listdir(dir):
            remove(f'{dir}/{file}')

        
    def __call__(self, step) -> None:
        self.counter += 1
        print(f'step {self.counter} : {len(step)}')
        open(f'{self.dir}/step-{self.counter}.{self.ext}', 'w').write(step)

def shuffleDict(_dict):
    keys = list(_dict.keys())
    shuffle(keys)
    new_dict = {key: _dict[key] for key in keys}
    return new_dict

"""combs = [
    ['l', 'I'],
    ['O', 'o'],
    ['N', 'M'],
    ['n', 'm'],
    ['S', 'Z'],
    ['U', 'V'],
    ['J', 'I'],
    ['j', 'i'],
    ['Q', 'O'],
    ['F', 'E'],
    ['d', 'b'],
    ['q', 'p'],
    ['D', 'O'],
    ['K', 'X']
]

comb = choice(combs)
shuffle(comb)
comb = {
    '0': comb[0],
    '1': comb[1]
}

def genRandomString():
    binary_id = bin(int(str(time_ns())[8:-3]))[2:]
    str_id = str.join('', [comb[x] for x in binary_id])
    return str_id"""

ucount = randint(3,10)
def genRandomString():
    global ucount
    name = '_' * ucount
    ucount += 1
    return name

class NameReplacer(ast.NodeTransformer):
    
    def __init__(
        self,
        _dict: dict = {},
        blacklist: list = []
    ) -> None:
        self._dict = _dict
        self.blacklist = blacklist
        
    def visit_Name(
        self,
        node: ast.Name
    ) -> ast.Name:
        
        if not node.id in self.blacklist:
            node.id = self._dict.get(node.id, node.id)
        
        return node
    
    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        node.attr = self.visit(ast.Name(id=node.attr, ctx=ast.Load())).id
        node.value = self.visit(node.value)
        return node
    
class AnnotationRemover(ast.NodeTransformer):
    def visit_arg(
        self,
        node: ast.arg
    ) -> ast.arg:
        node.annotation = None
        return node

    def visit_AnnAssign(
        self,
        node: ast.AnnAssign
    ) -> ast.Assign:
        return ast.Assign(targets=[node.target], value=node.value)

class VariableRenamer(NameReplacer):
    def visit_Name(
        self,
        node: ast.Name
    ) -> ast.Name:
        
        if not isinstance(node.ctx, ast.Load) and node.id not in self._dict:

            if not self._dict.get(node.id):
                newname = genRandomString()

                self._dict[node.id] = newname

        super().visit_Name(node)
        return node

    def visit_FunctionDef(
            self,
            node: ast.FunctionDef
    ) -> ast.FunctionDef:
        self.visit(ast.Module(body=node.body, type_ignores=[]))
        
        if not self._dict.get(node.name) and not (node.name.startswith('__') and node.name.endswith('__')):
            newname = genRandomString()

            self._dict[node.name] = newname
            node.name = newname

        else: node.name = self._dict.get(node.name, node.name)

        return node
    
    def visit_ClassDef(
            self,
            node: ast.ClassDef
    ) -> ast.ClassDef:
        self.visit(ast.Module(body=node.body, type_ignores=[]))
        
        if not self._dict.get(node.name) and not (node.name.startswith('__') and node.name.endswith('__')):
            newname = genRandomString()

            self._dict[node.name] = newname
            node.name = newname

        else: node.name = self._dict.get(node.name, node.name)

        return node
    
    def visit_arg(self, node: ast.arg) -> ast.arg:

        if not self._dict.get(node.arg):
            newname = genRandomString()

            self._dict[node.arg] = newname
            node.arg = newname
        else:
            node.arg = self._dict.get(node.arg, node.arg)

        return node
    
    def visit_Global(self, node: ast.Global) -> ast.Global:

        node.names = [self._dict.get(name, name) for name in node.names]

        return node

class JoinedStrFormatter(ast.NodeTransformer):
    def visit_JoinedStr(
        self,
        node: ast.JoinedStr
    ) -> ast.Call:
        print(node.__dict__)
        print([v.__dict__ for v in node.values])
        for v in node.values: print(v.value)
        return ast.Call(
            func=ast.Attribute(value=ast.Constant(value='{}'*len(node.values)), attr="format", ctx=ast.Load()),
            args=[value.value if isinstance(value, ast.FormattedValue) else value for value in node.values],
            keywords=[],
        )
    
class StrongStringObfuscator(ast.NodeTransformer):
    def visit_Str(
        self,
        node: ast.Str
    ) -> ast.Call:
                
        key = randint(5,50)

        x_name = genRandomString()
        y_name = genRandomString()
        
        return ast.Call(
            func=ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=x_name)],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=ast.Call(
                    func=ast.Attribute(value=ast.Name(id="str", ctx=ast.Load()), attr="join", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=""),
                        ast.Call(
                            func=ast.Name(id="map", ctx=ast.Load()),
                            args=[
                                ast.Name(id="chr", ctx=ast.Load()),
                                ast.Call(
                                    func=ast.Name(id="map", ctx=ast.Load()),
                                    args=[
                                        ast.Lambda(
                                            args=ast.arguments(
                                                posonlyargs=[],
                                                args=[ast.arg(arg=y_name)],
                                                kwonlyargs=[],
                                                kw_defaults=[],
                                                defaults=[],
                                            ),
                                            body=ast.BinOp(
                                                left=ast.Name(id=y_name, ctx=ast.Load()),
                                                op=ast.Sub(),
                                                right=ast.Constant(value=key),
                                            ),
                                        ),
                                        ast.Name(id=x_name, ctx=ast.Load()),
                                    ],
                                    keywords=[],
                                ),
                            ],
                            keywords=[],
                        ),
                    ],
                    keywords=[],
                ),
            ),
            args=[ast.Constant(value=bytes([ord(x)+key for x in node.s]))],
            keywords=[],
        )
        
class SoftStringObfuscator(ast.NodeTransformer):
    def visit_Str(
        self,
        node: ast.Str
    ) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="bytes", ctx=ast.Load()),
                    args=[ast.List(elts=[ast.Constant(value=byte) for byte in bytes(node.value.encode())], ctx=ast.Load())],
                    keywords=[],
                ),
                attr="decode",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        )

class BytesObfuscator(ast.NodeTransformer):
    def visit_Bytes(
        self,
        node: ast.Str
    ) -> ast.Call:
                
        key = randint(5, 255)
        
        return ast.Call(
            func=ast.Name(id="bytes", ctx=ast.Load()),
            args=[
                ast.Call(
                    func=ast.Name(id="map", ctx=ast.Load()),
                    args=[
                        ast.Lambda(
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[ast.arg(arg="x")],
                                kwonlyargs=[],
                                kw_defaults=[],
                                defaults=[],
                            ),
                            body=ast.BinOp(
                                left=ast.Name(id="x", ctx=ast.Load()),
                                op=ast.BitXor(),
                                right=ast.Constant(value=key),
                            ),
                        ),
                        ast.Constant(value=bytes(map(lambda x:x^key, node.s))),
                    ],
                    keywords=[],
                )
            ],
            keywords=[],
        )


class BuiltinObfuscator(ast.NodeTransformer):
    def visit_Name(
        self,
        node: ast.Name
    ) -> ast.Attribute:
        if hasattr(__builtins__, node.id) and not (node.id.startswith('__') and node.id.endswith('__')):
            return ast.Attribute(attr=node.id, value=ast.Name(id='__builtins__'))
        
        return node
        
class AttributeObfuscator(ast.NodeTransformer):
    def visit_Attribute(
        self,
        node: ast.Attribute
    ) -> ast.Call:
        
        if isinstance(node.ctx, ast.Store):
            return node
        
        return ast.Call(
            func=ast.Name(id='getattr', ctx=ast.Load()),
            args=[node.value, ast.Str(s=node.attr)],
            keywords=[]
        )
        
class ConstantObfuscator(ast.NodeTransformer):
    def __init__(self) -> None:
        
        self.class_name = genRandomString()
        self.consts = {}
    
    def visit_Constant(
        self,
        node: ast.Constant
    ) -> ast.Attribute:
        
        if node.value not in self.consts.values():
            alias_name = genRandomString()
            self.consts[alias_name] = node.value
        else:
            alias_name = [key for key, value in self.consts.items() if value == node.value][0]
        
        return ast.Attribute(value=ast.Name(id=self.class_name, ctx=ast.Load()), attr=alias_name, ctx=ast.Load())
    
    def constantClassDef(
        self,
        tree: ast.Module
    ):
        self.consts = shuffleDict(self.consts)
        
        constsclassdef = ast.Module(
            body=[
                ast.ClassDef(
                    name=self.class_name,
                    bases=[],
                    keywords=[],
                    body=[
                        ast.Assign(
                            targets=[
                                ast.Tuple(
                                    elts=[ast.Name(id=key, ctx=ast.Store()) for key in self.consts.keys()],
                                    ctx=ast.Store(),
                                )
                            ],
                            value=ast.Tuple(
                                elts=[ast.Constant(value=value) for value in self.consts.values()], ctx=ast.Load()
                            ),
                            lineno=1
                        )
                    ],
                    decorator_list=[],
                )
            ],
            type_ignores=[],
        )


        tree.body.insert(0, constsclassdef)
        return tree
    
class ImportObfuscator(ast.NodeTransformer):
    def visit_Import(
        self,
        node: ast.Import
    ) -> ast.Assign:
        
        return ast.Assign(
            targets=[ast.Tuple(elts=[ast.Name(id=name.name, ctx=ast.Store()) for name in node.names], ctx=ast.Store())],
            value=ast.Tuple(elts=[ast.Call(
                func=ast.Name(id="__import__", ctx=ast.Load()),
                args=[ast.Constant(value=name.name)],
                keywords=[],
            ) for name in node.names]),
            lineno=1
        )
    
    def visit_ImportFrom(
            self,
            node: ast.ImportFrom
        ):
        
        if ast.alias(name='*') in node.names:
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Call(func=ast.Name(id="vars", ctx=ast.Load()), args=[], keywords=[]),
                    attr="update",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Attribute(
                        value=ast.Call(
                            func=ast.Name(id="__import__", ctx=ast.Load()),
                            args=[ast.Constant(value="base64")],
                            keywords=[],
                        ),
                        attr="__dict__",
                        ctx=ast.Load(),
                    )
                ],
                keywords=[],
            )

        return ast.Assign(
            targets=[
                ast.Tuple(elts=[ast.Name(id=name.name, ctx=ast.Store()) for name in node.names], ctx=ast.Store())
            ],
            value=ast.ListComp(
                elt=ast.Call(
                    func=ast.Name(id="getattr", ctx=ast.Load()),
                    args=[ast.Call(func=ast.Name('__import__'), args=[ast.Constant(value=node.module)], keywords=[]), ast.Name(id="i", ctx=ast.Load())],
                    keywords=[],
                ),
                generators=[
                    ast.comprehension(
                        target=ast.Name(id="i", ctx=ast.Store()),
                        iter=ast.List(elts=[ast.Constant(value=name.name) for name in node.names], ctx=ast.Load()),
                        ifs=[],
                        is_async=0,
                    )
                ],
            ),
            lineno=1
        )

class EvalSerializer(ast.NodeTransformer):
    def serialize(self, node):
        return ast.Call(
            func=ast.Name(id='eval', ctx=ast.Load()),
            args=[ast.Constant(value=ast.unparse(node))],
            keywords=[]
        )

    def visit_DictComp(self, node: ast.DictComp) -> ast.Call:
        return self.serialize(node)
    
    def visit_ListComp(self, node: ast.ListComp) -> ast.Call:
        return self.serialize(node)
    
    def visit_SetComp(self, node: ast.SetComp) -> ast.Call:
        return self.serialize(node)
    
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.Call:
        return self.serialize(node)
        
    
class FunctionObfuscator(NameReplacer):
    
    def parse(
        self,
        code_obj: types.CodeType,
        typename: str,
        function_name: str
    ) -> ast.Call:
        namespace = {}
        _function = getattr(types, typename)(code_obj, namespace)
        
        _return = ast.Call(
                func=ast.Attribute(
                    value=ast.Call(
                        func=ast.Name(id="__import__", ctx=ast.Load()),
                        args=[ast.Constant(value="types")],
                        keywords=[],
                    ),
                    attr=typename,
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Call(
                                func=ast.Name(id="__import__", ctx=ast.Load()),
                                args=[ast.Constant(value="types")],
                                keywords=[],
                            ),
                            attr='CodeType',
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Constant(value=code_obj.co_argcount),
                            ast.Constant(value=code_obj.co_posonlyargcount),
                            ast.Constant(value=code_obj.co_kwonlyargcount),
                            ast.Constant(value=code_obj.co_nlocals),
                            ast.Constant(value=code_obj.co_stacksize),
                            ast.Constant(value=code_obj.co_flags),
                            ast.Constant(value=code_obj.co_code),
                            ast.parse(str(code_obj.co_consts)),
                            ast.parse(str(code_obj.co_names)),
                            ast.parse(str(code_obj.co_varnames)),
                            ast.Constant(value=code_obj.co_filename),
                            ast.Constant(value=function_name),
                            ast.Constant(value=code_obj.co_qualname),
                            ast.Constant(value=code_obj.co_firstlineno),
                            ast.Constant(value=code_obj.co_lnotab),
                            ast.Constant(value=code_obj.co_exceptiontable),
                            ast.parse(str(code_obj.co_freevars)),
                            ast.parse(str(code_obj.co_cellvars)),
                        ],
                        keywords=[]
                    ),
                    ast.Call(func=ast.Name(id="globals", ctx=ast.Load()), args=[], keywords=[]),
                    ast.Constant(value=''),
                    ast.parse(str(_function.__defaults__)),
                    ast.parse(str(_function.__closure__)),
                ],
                keywords=[]
            )
        
        ast.fix_missing_locations(_return)
        return _return
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.Assign:
        
        node.body = [super().visit(ast.Module(body=node.body, type_ignores=[]))]
        code_obj = next(filter(lambda x: isinstance(x, types.CodeType), compile(ast.unparse(node), '<string>', 'exec').co_consts))
        
        for const in code_obj.co_consts:
            if isinstance(const, types.CodeType):
                node.body = [self.visit(ast.Module(body=node.body, type_ignores=[]))]          
                code_obj = next(filter(lambda x: isinstance(x, types.CodeType), compile(ast.unparse(node), '<string>', 'exec').co_consts))
        
        parsed_node = self.parse(code_obj, 'FunctionType', node.name)
                
        decorators = node.decorator_list[::-1]
        
        for decorator in decorators:
            decorator = super().visit(decorator)
            parsed_node = ast.Call(func=decorator, args=[parsed_node], keywords=[])
            
        parsed_node = ast.Assign(
            targets=[ast.Name(id=node.name, ctx=ast.Store())],
            value=parsed_node,
            lineno=1
        )
        
        return parsed_node
    
    def visit_Lambda(self, node: ast.FunctionDef) -> ast.Assign:
        code_obj = compile(ast.unparse(node), '<string>', 'exec').co_consts[0]
        
        for const in code_obj.co_consts:
            if isinstance(const, types.CodeType):
                node.body = self.visit(ast.Module(body=[node.body], type_ignores=[]))
                code_obj = compile(ast.unparse(node), '<string>', 'exec').co_consts[0]
        
        return self.parse(code_obj, 'LambdaType', '')
    
class ClassObfuscator(NameReplacer):

    def parse(
            self,
            node: ast.ClassDef,
            class_name: str
    ):
        code_obj = next(filter(lambda x: isinstance(x, types.CodeType), compile(ast.unparse(node), '<string>', 'exec').co_consts))

        bases = [self._dict.get(base, base) for base in node.bases]

        localsspace_var_name = genRandomString()
        body_var_name = genRandomString() 

        return ast.Module(
            body=[
                ast.Assign(
                    targets=[
                        ast.Tuple(
                            elts=[
                                ast.Name(id=localsspace_var_name, ctx=ast.Store()),
                                ast.Name(id=body_var_name, ctx=ast.Store()),
                            ],
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Tuple(
                        elts=[
                            ast.Dict(keys=[], values=[]),
                            ast.Call(
                                func=ast.Attribute(
                                    value=ast.Call(
                                        func=ast.Name(id="__import__", ctx=ast.Load()),
                                        args=[ast.Constant(value="types")],
                                        keywords=[],
                                    ),
                                    attr="CodeType",
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.Constant(value=code_obj.co_argcount),
                                    ast.Constant(value=code_obj.co_posonlyargcount),
                                    ast.Constant(value=code_obj.co_kwonlyargcount),
                                    ast.Constant(value=code_obj.co_nlocals),
                                    ast.Constant(value=code_obj.co_stacksize),
                                    ast.Constant(value=code_obj.co_flags),
                                    ast.Constant(value=code_obj.co_code),
                                    ast.parse(str(code_obj.co_consts)),
                                    ast.parse(str(code_obj.co_names)),
                                    ast.parse(str(code_obj.co_varnames)),
                                    ast.Constant(value=code_obj.co_filename),
                                    ast.Constant(value=code_obj.co_name),
                                    ast.Constant(value=code_obj.co_qualname),
                                    ast.Constant(value=code_obj.co_firstlineno),
                                    ast.Constant(value=code_obj.co_lnotab),
                                    ast.Constant(value=code_obj.co_exceptiontable),
                                    ast.parse(str(code_obj.co_freevars)),
                                    ast.parse(str(code_obj.co_cellvars)),
                                ],
                                keywords=[],
                            ),
                        ],
                        ctx=ast.Load(),
                    ),
                    lineno=1
                ),
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="exec", ctx=ast.Load()),
                        args=[
                            ast.Name(id=body_var_name, ctx=ast.Load()),
                            ast.Call(
                                func=ast.Name(id="globals", ctx=ast.Load()),
                                args=[],
                                keywords=[],
                            ),
                            ast.Name(id=localsspace_var_name, ctx=ast.Load()),
                        ],
                        keywords=[],
                    )
                ),
                ast.Assign(
                    targets=[ast.Name(id=class_name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="type", ctx=ast.Load()),
                        args=[
                            ast.Constant(value=class_name),
                            ast.Tuple(elts=bases, ctx=ast.Load()),
                            ast.Name(id=localsspace_var_name, ctx=ast.Load()),
                        ],
                        keywords=[],
                    ),
                    lineno=1
                ),
            ],
            type_ignores=[],
        )



    def visit_ClassDef(self, node: ast.ClassDef) -> ast.Module:
        node.body = [super().visit(ast.Module(body=node.body, type_ignores=[]))]

        code_obj = compile(ast.unparse(node.body), '<string>', 'exec') # next(filter(lambda x: isinstance(x, types.CodeType), compile(ast.unparse(node), '<string>', 'exec').co_consts))

        for const in code_obj.co_consts:
            if isinstance(const, type):
                node.body = [self.visit(ast.Module(body=node.body, type_ignores=[]))]
                code_obj = next(filter(lambda x: isinstance(x, types.CodeType), compile(ast.unparse(node), '<string>', 'exec').co_consts))

        return self.parse(node, node.name)

class CodeCompiler(ast.NodeTransformer):
    def parse(
        self,
        code_obj: types.CodeType
    ):
        return ast.Module(
            body=[
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="exec", ctx=ast.Load()),
                        args=[
                            ast.Call(
                                func=ast.Attribute(
                                    value=ast.Call(
                                        func=ast.Name(id="__import__", ctx=ast.Load()),
                                        args=[ast.Constant(value="types")],
                                        keywords=[],
                                    ),
                                    attr="CodeType",
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.Constant(value=code_obj.co_argcount),
                                    ast.Constant(value=code_obj.co_posonlyargcount),
                                    ast.Constant(value=code_obj.co_kwonlyargcount),
                                    ast.Constant(value=code_obj.co_nlocals),
                                    ast.Constant(value=code_obj.co_stacksize),
                                    ast.Constant(value=code_obj.co_flags),
                                    ast.Constant(value=code_obj.co_code),
                                    ast.parse(str(code_obj.co_consts)),
                                    ast.parse(str(code_obj.co_names)),
                                    ast.parse(str(code_obj.co_varnames)),
                                    ast.Constant(value=code_obj.co_filename),
                                    ast.Constant(value=code_obj.co_name),
                                    ast.Constant(value=code_obj.co_qualname),
                                    ast.Constant(value=code_obj.co_firstlineno),
                                    ast.Constant(value=code_obj.co_lnotab),
                                    ast.Constant(value=code_obj.co_exceptiontable),
                                    ast.parse(str(code_obj.co_freevars)),
                                    ast.parse(str(code_obj.co_cellvars)),
                                ],
                                keywords=[],
                            )
                        ],
                        keywords=[],
                    )
                )
            ],
            type_ignores=[],
        )

    def visit(
        self,
        node: ast.Module
    ) -> ast.Module:
        code_obj = compile(ast.unparse(node.body), '<string>', 'exec')
        return self.parse(code_obj)

class BytesCompressor(ast.NodeTransformer):
    def visit_Bytes(
        self,
        node: ast.Bytes
    ):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="__import__", ctx=ast.Load()),
                    args=[ast.Constant(value="zlib")],
                    keywords=[],
                ),
                attr="decompress",
                ctx=ast.Load(),
            ),
            args=[ast.Constant(value=__import__("zlib").compress(node.s))],
            keywords=[],
        )

class CodeCompressor(ast.NodeTransformer):
    def visit(
        self,
        node: ast.Module
    ):
        return ast.Module(
            body=[
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="exec", ctx=ast.Load()),
                        args=[
                            ast.Call(
                                func=ast.Attribute(
                                    value=ast.Call(
                                        func=ast.Name(id="__import__", ctx=ast.Load()),
                                        args=[ast.Constant(value="zlib")],
                                        keywords=[],
                                    ),
                                    attr="decompress",
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.Constant(
                                        value=compress(ast.unparse(node).encode())
                                    )
                                ],
                                keywords=[],
                            )
                        ],
                        keywords=[],
                    )
                )
            ],
            type_ignores=[],
        )


class ClassPatcher(ast.NodeTransformer):
    def visit_ClassDef(
            self,
            node: ClassDef
        ) -> ast.ClassDef:

        

        return node


class Obfuscator:
    def __init__(
        self,
        source_code: str,
        rename_expressions: bool = True, # Rename variables, classes and functions
        string_obfuscation_level: int = False # Upgrade string obfuscation but can make the output program larger
    ):
        self.source_code: str = source_code
        self._code: str = source_code
        self.rename_expressions: bool = rename_expressions
        self.string_obfuscation_level: int = string_obfuscation_level
        
        step = StepFollower('steps', 'py')
        
        # layer 1

        self._code = minify(self._code)

        self._code = ast.unparse(JoinedStrFormatter().visit(self.tree))
        step(self._code)
        
        self._code = ast.unparse(ImportObfuscator().visit(self.tree))
        step(self._code)

        self._code = ast.unparse(AnnotationRemover().visit(self.tree))
        step(self._code)

        if self.rename_expressions:
            variable_renamer = VariableRenamer()
            self._code = ast.unparse(variable_renamer.visit(self.tree))
            self._code = ast.unparse(NameReplacer(variable_renamer._dict).visit(self.tree))
            step(self._code)
        
        self._code = ast.unparse(EvalSerializer().visit(self.tree))
        step(self._code)
        
        if string_obfuscation_level >= 1:
            self._code = ast.unparse(StrongStringObfuscator().visit(self.tree))
        else:
            self._code = ast.unparse(SoftStringObfuscator().visit(self.tree))
        step(self._code)

        self._code = ast.unparse(BytesObfuscator().visit(self.tree))
        step(self._code)
        
        self._code = ast.unparse(BuiltinObfuscator().visit(self.tree))
        step(self._code)

        self._code = ast.unparse(AttributeObfuscator().visit(self.tree))
        step(self._code)

        if string_obfuscation_level >= 2:
            self._code = ast.unparse(StrongStringObfuscator().visit(self.tree))
        else:
            self._code = ast.unparse(SoftStringObfuscator().visit(self.tree))
        step(self._code)
        
        self._code = ast.unparse(FunctionObfuscator().visit(self.tree))
        step(self._code)

        self._code = ast.unparse(ClassObfuscator().visit(self.tree))
        step(self._code)

        # layer 2

        self._code = minify(self._code)
        
        constant_obfuscator = ConstantObfuscator()
        self._code = ast.unparse(constant_obfuscator.constantClassDef(constant_obfuscator.visit(self.tree)))
        step(self._code)

        self._code = ast.unparse(ClassObfuscator().visit(self.tree))
        step(self._code)

        self._code = ast.unparse(AttributeObfuscator().visit(self.tree))
        step(self._code)

        self._code = ast.unparse(BuiltinObfuscator().visit(self.tree))
        step(self._code)

        #layer 3

        self._code = minify(self._code)

        self._code = ast.unparse(CodeCompiler().visit(self.tree))
        step(self._code)

        constant_obfuscator = ConstantObfuscator()
        self._code = ast.unparse(constant_obfuscator.constantClassDef(constant_obfuscator.visit(self.tree)))
        step(self._code)

        self._code = ast.unparse(ClassObfuscator().visit(self.tree))
        step(self._code)

        self._code = ast.unparse(BytesCompressor().visit(self.tree))
        step(self._code)

        self._code = ast.unparse(CodeCompressor().visit(self.tree))
        step(self._code)



        self._code = minify(self._code)




        
    @property
    def tree(self):
        return ast.parse(self._code)
        
    @property
    def out(self):
        return self._code

with open(file_path, 'r') as file:
    code = file.read()
    deoxys_obf_code = Obfuscator(code, rename_expressions=True, string_obfuscation_level=2).out
    name = str.join('.', file.name.split('.')[:-1])
    ext = '.' + file.name.split('.')[-1]
    open(name+'-obf'+ext, 'w', encoding='utf8').write(deoxys_obf_code)