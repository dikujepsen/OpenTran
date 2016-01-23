class A(object):
    def foo(self):
        print self._foo1()

    def _foo1(self):
        return "foo"

class B(A):
    def _foo1(self):
        print "B-foo1"

#    def foo(self):
#        raise AttributeError, "wtf!"

    def foo2(self):
        self.foo()

myB = B()
myB.foo2()