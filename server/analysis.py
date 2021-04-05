import lzstring
import asyncio
import json
import numpy as np
import matplotlib.pyplot as plt
from databases import Database
import os
from matplotlib.patches import Rectangle

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

dbpath = "/data.db"
if os.getcwd().endswith("/server"):
    print("Sending data to local database")
    dbpath = "../../../data.db"

database = Database("sqlite://" + dbpath)

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
class StraightMachine(EntityType):
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
class Merger(EntityType):
    def __init__(self, entity, json_data):
        super().__init__(entity, json_data)
        self.flipped = 1 if json_data["components"]["StaticMapEntity"]["code"] == 6 else -1
        self.outputs = [(-self.dir, -self.dir)]
        self.inputs = [(self.flipped * J(self.dir), self.flipped * J(self.dir)), (self.dir, self.dir)]
# The following machines are approximated as belts: 49 = Belt reader
codes = {1:Belt, 2: BeltTurn, 3: BeltTurn, 4:Balancer, 6: Merger, 7: Miner, 8: Miner, 9: Splitter, 11: StraightMachine, 14: Stacker, 15: Mixer, 16: Painter, 17: Painter, 22: Tunnel, 23: Tunnel, 24: Tunnel, 25: Tunnel, 26:Hub, 49: Belt}

def search(grid, start, end):
    for x in range(start[0], end[0]):
        for y in range(start[1], end[1]):
            if (x, y) in grid:
                yield grid[(x, y)]

SEARCH_RADIUS = 10
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
    def code(self):
        return (self.typecode, int(self.dir[0]), int(self.dir[1]))
    def matches(self, other):
        return n
    def identify_repeats(self, grid, map):
        self.repeats = {}
        # todo: quicker search using trees + separating by type. or something.
        for entity in search(grid, (self.x - SEARCH_RADIUS, self.y - SEARCH_RADIUS), (self.x + SEARCH_RADIUS + 1, self.y + SEARCH_RADIUS + 1)):
            if not entity is self:
                off_x = entity.x - self.x
                off_y = entity.y - self.y
                i = 2
                while (self.x + i * off_x, self.y + i * off_y) in grid:
                    i = i + 1
                i = i - 1
                j = -1
                while (self.x + j * off_x, self.y + j * off_y) in grid:
                    j = j - 1
                j = j + 1
                # ok so now we know that there's repeats relative to this machine i times in
                # the direction of (off_x, off_y), and j times in the opposite direction...
                self.repeats[(off_x, off_y)] = (i, j)
    def identify_module(self, map, init_used):
        best = None
        best_size = 0
        best_rep = 0
        best_off = None
        best_used = None
        for off, (up, down) in self.repeats.items():
            # we'll start from one end of the module, so I guess we'll ignore 'down'; maybe I shouldn't
            # have computed it in the first place...

            # it's kind of inefficient to do this as a loop; it seems like we should've been able
            # to dynamically shrink things as relevant. but oh, whatever.
            for alt_up in range(up, 0, -1):
                relevant = self.x < -80 and self.y < -62 and self.x > -115 and self.y > -69 and alt_up > 8
                queue = set([self])
                module = set()
                used = set()
                while len(queue) > 0:
                    potential_piece = queue.pop()
                    if potential_piece.repeats[off][0] >= alt_up:
                        skip = False
                        # basically what can go wrong is that we try adding some piece whose repetitions are already used
                        # ... so we need to check whether all of its repetitions are unused. or maybe this can be made faster/cleverer
                        for i in range(0, alt_up+1):
                            it = map.grid[potential_piece.x + i * off[0], potential_piece.y + i * off[1]]
                            if it in used or it in init_used:
                                skip = True
                                break
                        if skip:
                            continue
                        module.add(potential_piece)
                        for i in range(0, alt_up+1):
                            used.add(map.grid[potential_piece.x + i * off[0], potential_piece.y + i * off[1]])
                        # todo take connection compatibility into account
                        if potential_piece.type is not None:
                            for nx, ny in potential_piece.neighbors():
                                ent = map.grid.get((nx, ny))
                                if not ent is None and not ent in used and off in ent.repeats:
                                    queue.add(ent)
                # todo prune dangling belts
                size = len(module) * (alt_up - 1) # magic formula for trading off internal size vs # of repeats
                if size > best_size:
                    best = module
                    best_size = size
                    best_rep = alt_up # todo is this actually counting right
                    best_off = off
                    best_used = used
        return best, best_size, best_rep, best_off, best_used
    def neighbors(self, dir=None):
        if self.type is None:
            print(self.shape)
            for xo, yo in self.shape:
                print(xo, yo)
            neighbors = [ (x+xo, y+yo)
                for x, y in self.shape
                for xo in [-1, 0, 1] for yo in [-1, 0, 1]
                if xo**2 + yo**2 == 1
                if not (x+xo, y+yo) in [list(x) for x in self.shape] # todo ugly hack to bypass numpy broadcasting
            ]
        else:
            consider = ([] if dir == "up" else self.type.inputs) + ([] if dir == "down" else self.type.outputs)
            neighbors = [ (self.x+xo, self.y+yo) for ((xo, yo), _) in consider]
        return neighbors
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
        while len(to_unmark) > 0:
            unmarking = to_unmark.pop()
            shouldPrint = unmarking.x == 0 and unmarking.y > 0 and unmarking.y < 11
            if not unmarking in unconnected:
                continue
            unconnected.remove(unmarking)
            neighbors = unmarking.neighbors(dir=direction)
            neighbors = [ (x, y)
                for (x, y) in neighbors
                if (x, y) in self.grid
            ]
            for x, y in neighbors:
                to_unmark.add(self.grid[x, y])
        return unconnected
    def mark_obsolete(self):
        obsolete_downstream = self.mark_unconnected([self.hub], "down")
        # consider implementing something that checks connection to all required resources, rather than just some
        obsolete_upstream = self.mark_unconnected([entity for entity in self.entities if type(entity.type) is Miner], "up")
        self.obsolete = obsolete_downstream.union(obsolete_upstream)
        return [(item.x, item.y) for item in self.obsolete]
    def identify_best_module(self, used):
        best = None
        best_size = 0
        best_rep = 0
        best_off = None
        best_used = set()
        for entity in self.entities:
            # belts should not be a core part of modules...
            if entity.typecode in [1, 2, 3]:
                continue
            module, module_size, module_repeats, module_off, module_used = entity.identify_module(self, used)
            if module_size > best_size:
                best = module
                best_size = module_size
                best_rep = module_repeats
                best_off = module_off
                best_used = module_used
        return best, best_size, best_rep, best_off, best_used
    def identify_modules(self):
        # for faster search & simpler code, split the grid by machine type
        subgrids = {}
        for entity in self.entities:
            if not entity.code() in subgrids:
                # todo use octree
                subgrids[entity.code()] = {}
            subgrids[entity.code()][(entity.x, entity.y)] = entity
        print(subgrids.keys())
        for entity in self.entities:
            entity.identify_repeats(subgrids[entity.code()], self)
        used = set([self.hub])
        modules = [Module(set(used), 1, (0, 0), set(used))]
        while True:
            module, module_size, module_rep, module_off, module_used = self.identify_best_module(used)
            if module_rep < 2 or len(module) < 3:
                break
            used.update(module_used)
            modules.append(Module(module, module_rep, module_off, module_used))
            #if len(modules) > 1:
            #    break
        # kinda hacky maybe, but the other algorithms assume that the entire map is split into modules,
        # and the previous loop does not do so for small/lone machines. so we instead just turn every
        # lonesome machine into a module
        for entity in self.entities:
            if not entity in used and not entity.typecode in [1, 2, 3, 22, 23, 24, 25]:
                used.add(entity)
                modules.append(Module(set([entity]), 1, (0, 0), set([entity])))
        self.modules = modules
        return modules, used
    def trace_belt(self, belt):
        path = []
        while belt is not None and type(belt.type) in [Belt, BeltTurn, Tunnel]:
            path.append(belt)
            if len(belt.type.inputs) != 1:
                break # this is a tunnel that has not been connected to other stuff
            (xo, yo), _ = belt.type.inputs[0]
            belt = self.grid.get((belt.x + xo, belt.y + yo)) # todo take connectedness into account better
        n_turns = sum([1 if type(b.type) is BeltTurn else 0 for b in path[1:-1]]) # remove endpoints because turns are allowed there
        n_turns = max(n_turns - 1, 0) # a single turn is needed to cover most of the plane
        return path, n_turns
    def identify_connecting_paths(self):
        used = set()
        all_belt_turns = []
        for module in self.modules:
            for machine in module.machines:
                for position in machine.neighbors():
                    ent = self.grid.get(position)
                    if ent is None:
                        continue # while this machine has a connection point here, it does not connect to an entity
                    if ent in module.machines:
                        continue # this is just an internal connection within the module
                    if type(ent.type) not in [Belt, BeltTurn, Tunnel]:
                        continue # this is a connection to a non-belt machine; aka not a connecting path
                    if len(ent.type.outputs) != 1:
                        continue # this is a tunnel that has not been connected to other stuff
                    if ent in used: # todo if we just trace the paths upward is that even necessary to consider?
                        continue # this belt has already been identified
                    (xo, yo), _ = ent.type.outputs[0]
                    if self.grid.get((ent.x + xo, ent.y + yo)) != machine:
                        continue # this belt is either an input belt receiving items from the module, or just not properly connected to the module
                    belt_path, n_turns = self.trace_belt(ent)
                    used.update(belt_path)
                    all_belt_turns.append(n_turns)
        return used, sum(all_belt_turns)/len(all_belt_turns)
class Module:
    def __init__(self, configuration, repeats, offset, machines):
        self.bb = (
            (min([pt[0] for m in machines for pt in m.shape]),
             min([pt[1] for m in machines for pt in m.shape]))
            ,
            (max([pt[0] for m in machines for pt in m.shape]),
             max([pt[1] for m in machines for pt in m.shape]))
        )
        self.width = self.bb[1][0] - self.bb[0][0]
        self.height = self.bb[1][1] - self.bb[0][1]
        self.configuration = configuration
        self.machines = machines
async def main():
    await database.connect()
    rows = await database.fetch_all(query="SELECT * FROM Saves")
    n_obs = []
    select = None
    for i, (userId, storeTime, compressed_save_data) in enumerate(rows):
        decompressed_data = decompress_save_data(compressed_save_data)
        map = MapState(decompressed_data)
        obsolete = np.array(map.mark_obsolete())
        modules, machines_in_modules = map.identify_modules()
        paths, n_turns = map.identify_connecting_paths()
        n_obs.append(len(obsolete))
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
        config = []
        for entity in map.entities:
            if entity in machines_in_modules:
                continue
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
                for xo in [-1, 0, 1]:
                    for yo in [-1, 0, 1]:
                        if xo**2 + yo**2 == 1 and map.grid.get((x+xo, y+yo)) is entity:
                            shp.append(np.array([x+0.33*xo, y + 0.33*yo]))
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
        fig, ax = plt.subplots()
        plt.scatter(pts[:, 0], -pts[:, 1], c="#1f77b4")
        plt.scatter(opts[:, 0], -opts[:, 1], c="#ff7f0e")
        plt.scatter(ipts[:, 0], -ipts[:, 1], c="#2ca02c")
        plt.scatter(shp[:, 0], -shp[:, 1], c="#d62728")
        plt.scatter(obsolete[:, 0], -obsolete[:, 1], c="#9467bd")
        for i, module in enumerate(modules):
            ax.add_patch(Rectangle([module.bb[0][0], -module.bb[0][1] - module.height], module.width, module.height, linewidth=1, edgecolor='b', facecolor='none'))
            config = np.array([pt for m in module.configuration for pt in m.shape]).reshape((-1, 2))
            plt.scatter(config[:, 0], -config[:, 1], c="#404040")
            plt.text(module.bb[0][0], -module.bb[0][1], str(i))
        paths = np.array([pt for m in paths for pt in m.shape]).reshape((-1, 2))
        plt.scatter(paths[:, 0], -paths[:, 1], c="#FFFF00")
        #for i in range(pts.shape[0]):
        #    plt.text(pts[i, 0], -pts[i, 1], str(tcodes[i]))
        plt.xlim(-150, 150)
        plt.ylim(-150, 150)
        plt.show()
        print(userId + " @ " + storeTime + ": " + str(len(json.dumps(decompressed_data))) + " from " + str(len(compressed_save_data)))
        print(n_turns)

asyncio.get_event_loop().run_until_complete(main())