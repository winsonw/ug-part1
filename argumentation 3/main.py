import web

urls = (
    "/", "WebIndex",
    "/addArg", "WebAddArg",
    "/addArgument", "WebAddArgument",
    "/addRelation", "WebAddRelation"
    "/resetAll", "WebResetAll",
    "/outputAll", "WebOutputAll"

)

app = web.application(urls, globals())
render=web.template.render('templates')


class Argument:
    def __init__(self, index, description):
        self.id = index
        self.description = description
        self.attack_to = []
        self.attack_from = []
        self.is_in = False
        self.is_out = False
        self.is_free = True

    def add_attack(self, index):
        self.attack_to.append(index)

    def add_been_attacked(self, index):
        self.attack_from.append(index)

    def remove_been_attack(self, index):
        self.attack_from.remove(index)

    def become_in(self):
        if len(self.attack_from) == 0:
            self.is_in = True
            self.is_free = False
            return True
        return False

    def become_out(self):
        self.is_out = True
        self.is_free = False

    def __repr__(self):
        return str(self.id)

class Algorithm:
    def __init__(self):
        self.resetAll()

        #self.read_data()

    def read_data(self,prosCount):
        # print('IN :')
        # prosCount = input()
        # print("You.... IN :", prosCount)

        self.arguments = [Argument(i,i) for i in range(int(prosCount))]
        pre_relations = [(1,2),(2,3),(4,5),(5,4)]
        for relation in pre_relations:
            attack = relation[0]-1
            attacked = relation[1]-1
            self.relations.append((attack, attacked))
            self.arguments[attack].add_attack(attacked)
            self.arguments[attacked].add_been_attacked(attack)

    def resetAll(self):
         self.arguments =[]
         self.relations=[]
         self.ins=[]
         self.outs=[]

    def addArg(self,id,description,attackId):

        argument= Argument(id,description)
        argument.add_attack(attackId)

        self.arguments.append(argument)
        self.relations.append((id,attackId))
        self.arguments[attackId].add_been_attacked(id)

        # print("IN :"+strIN)
        # print("OUT:"+strOUT)

        # print("IN :"+strIN)
        # print("OUT:"+strOUT)

    def indexProcess(self, index):
        return index-1

    def addArgument(self, index, description):
        index = self.indexProcess(index)

        argument= Argument(index,description)
        self.arguments.append(argument)

        self.learn()

    def addRelation(self, attackId, attackedId):
        attackId = self.indexProcess(attackId)
        attackedId = self.indexProcess(attackedId)

        self.relations.append((attackId,attackedId))
        self.arguments[attackId].add_attack(attackedId)
        self.arguments[attackedId].add_attack(attackId)

        self.learn()
        

    def learn(self):
        new_in_count = len(self.ins)
        new_out_count = len(self.outs)

        old_in_count = -1
        old_out_count = -1

        while (old_in_count != new_in_count) or (old_out_count != new_out_count):
            old_in_count = new_in_count
            old_out_count = new_out_count

            self.find_in()

            new_in_count = len(self.ins)
            new_out_count = len(self.outs)


        strIN = ''.join(','+str(i) for i in self.ins)
        strOUT = ''.join(','+str(i) for i in self.outs)
        return "IN :"+strIN+" OUT:"+strOUT
        # print("IN :"+strIN)
        # print("OUT:"+strOUT)

    def find_in(self):
        for argument in self.arguments:
            if argument.is_free:
                if argument.become_in():
                    self.ins.append(argument)
                    for attack in argument.attack_to:
                        self.arguments[attack].become_out()
                        self.outs.append(attack)
                        for remove_attack in self.arguments[attack].attack_to:
                            self.arguments[remove_attack].remove_been_attack(attack)


algorithm_MM = Algorithm()

class WebOutputAll:
    def GET(self):
        res=algorithm_MM.learn()
        web.header('Content-Type', 'application/javascript;charset=utf-8')
        return res

class WebResetAll:
    def GET(self):

        algorithm_MM.resetAll()
        res ="Reset OK"
        web.header('Content-Type', 'application/javascript;charset=utf-8')
        return res


class WebAddArg:

    def GET(self):
        i = web.input()
        fromArg = i.fromArg
        toArg = i.toArg
        desc = i.desc

        print ('fromArg='+str(fromArg)+' toArg='+toArg+'  desc='+desc)
        algorithm_MM.addArg(int(fromArg),desc,int(toArg))
        res ="Add Arg OK"
        web.header('Content-Type', 'application/javascript;charset=utf-8')
        return res


class WebAddArgument:

    def GET(self):
        i = web.input()
        fromArg = i.fromArg
        desc = i.desc

        print ('fromArg='+str(fromArg)+'  desc='+desc)
        algorithm_MM.addArgument(int(fromArg),desc)
        res ="Add Argument OK"
        web.header('Content-Type', 'application/javascript;charset=utf-8')
        return res

class WebAddRelation:

    def GET(self):
        i = web.input()
        fromArg = i.fromArg
        toArg = i.toArg

        print ('fromArg='+str(fromArg)+' toArg='+toArg)
        algorithm_MM.addRelation(int(fromArg),int(toArg))
        res ="Add Relation OK"
        web.header('Content-Type', 'application/javascript;charset=utf-8')
        return res


class WebIndex:
    def GET(self):
        render=web.template.frender("templates/index.html")
        return render()

if __name__ == "__main__":
    web.internalerror = web.debugerror
    algorithm_MM = Algorithm()
    app.run()
