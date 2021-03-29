import lzstring
import asyncio
import json
import numpy as np
import matplotlib.pyplot as plt
from databases import Database

compression = lzstring.LZString()
charmap = "!#%&'()*+,-./:;<=>?@[]^_`{|}~¥¦§¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿABCDEFGHIJKLMNOPQRSTUVWXYZ"

def decompress_int(strForm):
    result = 0
    for ch in strForm:
        result = result * len(charmap) + charmap.index(ch)
    result -= 1
    return result

def decompress_object(obj, keys, values):
    if type(obj) is list:
        return [decompress_object(element, keys, values) for element in obj]
    elif type(obj) is dict:
        return {keys[decompress_int(key)]:decompress_object(value, keys, values) for key, value in obj.items()}
    elif type(obj) is str:
        return values[decompress_int(obj)]
    else:
        return obj

def decompress_save_data(save_data):
    save_data = compression.decompressFromEncodedURIComponent(save_data[1:])[40:]
    save_data = json.loads(save_data)
    save_data = decompress_object(save_data["data"], save_data["keys"], save_data["values"])
    return save_data

database = Database("sqlite:///data.db")

def print_tree_structure(value, depth=0):
    if type(value) is dict:
        for key in value.keys():
            print("\t"*depth + key)
            print_tree_structure(value[key], depth+1)
    elif type(value) is list:
        print("\t"*depth + "[list] e.g.")
        if len(value) > 0:
            print_tree_structure(value[0], depth+1)
    elif type(value) is int:
        value = str(value)
    if type(value) is str:
        print("\t"*depth + value)

rotation_dirs = [(0, 1), (-1, 0), (0, -1), (1, 0)]
rotation_dirs = [np.array(pt) for pt in rotation_dirs]
def J(vec):
    return np.array([-vec[1], vec[0]])
#rotation_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class EntityType:
    def __init__(self, entity, json_data):
        self.w = 1
        self.h = 1
        self.dir = rotation_dirs[json_data["components"]["StaticMapEntity"]["rotation"]//90]
        # list of (offset, direction)
        self.inputs = []
        self.outputs = []
class Hub(EntityType):
    def __init__(self, entity, json_data):
        super().__init__(entity, json_data)
        self.w = 4
        self.h = 4
        self.inputs = [(np.array((x, y)), np.array(d)) for i in range(4) for x, y, d in [(-1, i, (-1, 0)), (4, i, (1, 0)), (i, -1, (0, -1)), (i, 4, (0, 1))]]
class Miner(EntityType):
    def __init__(self, entity, json_data):
        super().__init__(entity, json_data)
        self.outputs = [(-self.dir, -self.dir)]
        self.inputs = [(-J(self.dir), -J(self.dir)), (self.dir, self.dir), (J(self.dir), J(self.dir))] # approximation to deal with chain miners
        print(self.outputs)
        pass
class Splitter(EntityType):
    def __init__(self, entity, json_data):
        super().__init__(entity, json_data)
        self.inputs = [(self.dir, self.dir)]
        self.outputs = [(-self.dir, -self.dir), (-self.dir - J(self.dir), -self.dir)]
        self.h = 2
class Balancer(EntityType):
    def __init__(self, entity, json_data):
        super().__init__(entity, json_data)
        self.inputs = [(self.dir, self.dir), (self.dir - J(self.dir), self.dir)]
        self.outputs = [(-self.dir, -self.dir), (-self.dir - J(self.dir), -self.dir)]
        self.h = 2
class Tunnel(EntityType):
    def __init__(self, entity, json_data):
        super().__init__(entity, json_data)
        self.w = 1
        self.h = 1
        self.max_dist = 6 if json_data["components"]["StaticMapEntity"]["code"] in [22, 23] else 10 # todo upgrades? :thinking:
    def connect(self, entry, exit):
        self.outputs = [(np.array([exit.x - entry.x, exit.y - entry.y]) - exit.type.dir, -exit.type.dir)]
        self.inputs = [(entry.type.dir, entry.type.dir)]
        entry.shape.append((exit.x, exit.y))
        entry.w = None; self.w = None
        entry.h = None; self.h = None
class Belt(EntityType):
    def __init__(self, entity, json_data):
        super().__init__(entity, json_data)
        self.inputs = [(self.dir, self.dir)]
        self.outputs = [(-self.dir, -self.dir)]
class BeltTurn(EntityType):
    def __init__(self, entity, json_data):
        super().__init__(entity, json_data)
        self.curve = "right" if json_data["components"]["StaticMapEntity"]["code"] == 3 else "left"
        self.inputs = [(self.dir, self.dir)]
        if self.curve == "right":
            self.outputs = [(-J(self.dir), -J(self.dir))]
        else:
            self.outputs = [(J(self.dir), J(self.dir))]
class Painter(EntityType):
    def __init__(self, entity, json_data):
        super().__init__(entity, json_data)
        self.flipped = 1 if json_data["components"]["StaticMapEntity"]["code"] == 17 else -1
        self.h = 2
        self.outputs = [(-2*J(self.dir), -J(self.dir))]
        self.inputs = [(J(self.dir), J(self.dir)), (self.flipped * self.dir - J(self.dir), self.flipped * self.dir)]
class Mixer(EntityType):
    def __init__(self, entity, json_data):
        super().__init__(entity, json_data)
        self.inputs = [(self.dir, self.dir), (self.dir - J(self.dir), self.dir)]
        self.outputs = [(-self.dir, -self.dir)]
        self.h = 2
class Stacker(EntityType):
    def __init__(self, entity, json_data):
        super().__init__(entity, json_data)
        self.inputs = [(self.dir, self.dir), (self.dir - J(self.dir), self.dir)]
        self.outputs = [(-self.dir, -self.dir)]
        self.h = 2
# The following machines are approximated as belts: 49 = Belt reader, 11 = Rotator
codes = {1:Belt, 2: BeltTurn, 3: BeltTurn, 4:Balancer, 7: Miner, 8: Miner, 9: Splitter, 11: Belt, 14: Stacker, 15: Mixer, 16: Painter, 17: Painter, 22: Tunnel, 23: Tunnel, 24: Tunnel, 25: Tunnel, 26:Hub, 49: Belt}

class Entity:
    def __init__(self, json_data, x, y):
        self.x = x
        self.y = y
        self.dir = rotation_dirs[json_data["components"]["StaticMapEntity"]["rotation"]//90]
        entity_type = codes.get(json_data["components"]["StaticMapEntity"]["code"])
        if not entity_type is None:
            self.type = entity_type(self, json_data)
        else:
            self.type = None
        self.typecode = json_data["components"]["StaticMapEntity"]["code"]
        w, h = 1, 1
        if not self.type is None:
            w = self.type.w
            h = self.type.h
        self.w = w
        self.h = h
        self.shape = [np.array((x, y)) + self.dir * xo - J(self.dir) * yo for xo in range(w) for yo in range(h)]
    def add_to_map(self, map_state):
        for x, y in self.shape:
            map_state.grid[x, y] = self
class MapState:
    def __init__(self, json_data):
        self.grid = {}
        self.entities = []
        self.tunnel_entries = {}
        self.tunnel_exits = {}
        for entity in json_data["dump"]["entities"]:
            if "StaticMapEntity" in entity["components"]:
                SMP = entity["components"]["StaticMapEntity"]
                x, y = SMP["origin"]["x"], SMP["origin"]["y"]
                self.entities.append(Entity(entity, x, y))
                if type(self.entities[-1].type) is Hub:
                    self.hub = self.entities[-1]
                if type(self.entities[-1].type) is Tunnel:
                    # this is a tunnel, we need to connect start point/end point
                    if SMP["code"] in [22, 24]:
                        self.tunnel_entries[x, y] = self.entities[-1]
                    else:
                        self.tunnel_exits[x, y] = self.entities[-1]
        # connect tunnels
        for (nx, ny), entry in self.tunnel_entries.items():
            print("trying tunnel " + str((nx, ny)))
            dir = entry.type.dir
            best = None
            for i in range(entry.type.max_dist):
                tx, ty = nx - i*dir[0], ny - i*dir[1]
                print("walk " + str((tx, ty)))
                if (tx, ty) in self.tunnel_exits:
                    if not np.all(self.tunnel_exits[(tx, ty)].type.dir == dir):
                        best = ((tx, ty), self.tunnel_exits[(tx, ty)].type.dir)
                        continue
                    break
            else:
                print("missing tunnel at " + str((nx, ny)) + " in dir " + str(dir) + " (best: " + str(best) + ")")
                continue
            print()
            exit = self.tunnel_exits[tx, ty]
            self.entities = [entity for entity in self.entities if not entity is exit] # todo make faster
            entry.type.connect(entry, exit)
        for entity in self.entities:
            entity.add_to_map(self)
    def mark_unconnected(self, base, direction):
        # todo: take connectedness and connection direction into account
        unconnected = set(self.entities)
        to_unmark = set(base)
        iteration = 0
        shouldPrint = False
        while len(to_unmark) > 0:
            if shouldPrint:
                print([(e.x, e.y) for e in to_unmark])
            unmarking = to_unmark.pop()
            shouldPrint = unmarking.x == 0 and unmarking.y > 0 and unmarking.y < 11
            if not unmarking in unconnected:
                continue
            if shouldPrint:
                print(unmarking.x, unmarking.y)
            unconnected.remove(unmarking)
            neighbors = [ (x+xo, y+yo)
                for x, y in unmarking.shape
                for xo in [-1, 0, 1] for yo in [-1, 0, 1]
                if xo**2 + yo**2 == 1
            ]
            if shouldPrint:
                print(neighbors)
            if not unmarking.type is None:
                if direction == "down":
                    neighbors = [ (unmarking.x + x, unmarking.y + y) for (x, y), dir in unmarking.type.inputs ]
                if direction == "up":
                    neighbors = [ (unmarking.x + x, unmarking.y + y) for (x, y), dir in unmarking.type.outputs ]
            if shouldPrint:
                print(neighbors)
            neighbors = [ (x, y)
                for (x, y) in neighbors
                if (x, y) in self.grid
                if not self.grid[x, y] is unmarking
            ]
            if shouldPrint:
                print(neighbors)
            for x, y in neighbors:
                to_unmark.add(self.grid[x, y])
                if shouldPrint:
                    ent = self.grid[x, y]
                    print((ent.x, ent.y))
            #iteration += 1
            #if iteration > 40000:
            #    break
            if shouldPrint:
                print()
        return unconnected
    def mark_obsolete(self):
        obsolete_downstream = self.mark_unconnected([self.hub], "down")
        print(len(obsolete_downstream))
        obsolete_upstream = self.mark_unconnected([entity for entity in self.entities if type(entity.type) is Miner], "up")
        print(len(obsolete_upstream))
        self.obsolete = obsolete_downstream.union(obsolete_upstream)
        print(len(self.obsolete))
        return [(item.x, item.y) for item in self.obsolete]

async def main():
    await database.connect()
    rows = await database.fetch_all(query="SELECT * FROM Saves")
    n_obs = []
    # 182 = cut off
    # 183 = fine again
    select = None
    for i, (userId, storeTime, compressed_save_data) in enumerate(rows):
        if i < 178 or i > 188 or (select != None and i != select):
            continue
        decompressed_data = decompress_save_data(compressed_save_data)
        map = MapState(decompressed_data)
        obsolete = np.array(map.mark_obsolete())
        n_obs.append(len(obsolete))
        if i == select:
            break
    plt.plot(np.arange(len(n_obs)), n_obs)
    plt.xlabel("save # (could also have chosen playtime)")
    plt.ylabel("# of obsolete machines")
    plt.show()
    if True:
        if False:
            plt.scatter(obsolete[:, 0], -obsolete[:, 1])
            plt.xlim(-20, 20)
            plt.ylim(-20, 20)
            plt.show()
        pts = []
        tcodes = []
        shp = []
        ipts = []
        opts = []
        for entity in map.entities:
            #if type(entity.type) is not Balancer:
            #    continue
            #if entity.x < -10 or entity.y > 10 or entity.y < -10 or entity.y > 10:
            #    continue
            pos = np.array([entity.x, entity.y])
            pts.append(pos)
            tcodes.append(entity.typecode)
            for x, y in entity.shape:
                if x != entity.x or y != entity.y:
                    shp.append(np.array([x, y]))
            if entity.type != None:
                for o, d in entity.type.outputs:
                    opts.append(o+pos - 0.66*d)
                for i, d in entity.type.inputs:
                    ipts.append(i+pos - 0.66*d)
            #break
        pts = np.array(pts).reshape((-1, 2))
        ipts = np.array(ipts).reshape((-1, 2))
        opts = np.array(opts).reshape((-1, 2))
        shp = np.array(shp).reshape((-1, 2))
        plt.scatter(pts[:, 0], -pts[:, 1], c="#1f77b4")
        plt.scatter(opts[:, 0], -opts[:, 1], c="#ff7f0e")
        plt.scatter(ipts[:, 0], -ipts[:, 1], c="#2ca02c")
        plt.scatter(shp[:, 0], -shp[:, 1], c="#d62728")
        plt.scatter(obsolete[:, 0], -obsolete[:, 1], c="#9467bd")
        #for i in range(pts.shape[0]):
        #    plt.text(pts[i, 0], -pts[i, 1], str(tcodes[i]))
        plt.xlim(-150, 150)
        plt.ylim(-150, 150)
        plt.show()
        print(userId + " @ " + storeTime + ": " + str(len(json.dumps(decompressed_data))) + " from " + str(len(compressed_save_data)))

asyncio.get_event_loop().run_until_complete(main())