import lzstring
import asyncio
import json
import numpy as np
import matplotlib.pyplot as plt
from databases import Database
import os
from matplotlib.patches import Rectangle
import dateutil.parser as dp
import analysis
import csv
from sklearn.svm import SVR
from factor_analyzer import FactorAnalyzer
from sklearn.linear_model import LinearRegression

dbpath = "/data.db"
if os.getcwd().endswith("/server"):
    print("Sending data to local database")
    dbpath = "../../../data.db"

database = Database("sqlite://" + dbpath)

def plot_loghist(x, bins=10):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins)
    plt.xscale('log')
def ax_loghist(ax, x, bins=10):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    ax.hist(x, bins=logbins)
    ax.set_xscale('log')

async def handle_save(user_id, path, storeTime, save_data, before, after):
    last_map = analysis.MapState(save_data)
    obsolete = np.array(last_map.mark_obsolete())
    modules, machines_in_modules = last_map.identify_modules()
    paths, n_turns = last_map.identify_connecting_paths()
    with open(path + "/" + storeTime + ".html", "w") as f:
        f.write("<html><head><title>Save</title></head>")
        f.write("<body>")
        f.write("<h1>Save at time" + storeTime + " for player " + user_id + "</h1>")
        f.write("<p>")
        if not before is None:
            f.write("<a href=\"" + before + ".html\">&lt;&lt;</a> ")
        f.write("<a href=\"index.html\">^^</a>")
        if not after is None:
            f.write(" <a href=\"" + after + ".html\">&gt;&gt;</a>")
        f.write("</p>")
        old_sz = plt.rcParams["figure.figsize"]
        plt.rcParams["figure.figsize"] = (last_map.w * 0.33 + 1,last_map.h * + 0.33 + 1)
        analysis.render_plt(last_map)
        plt.savefig(path + "/" + storeTime + ".png")
        plt.close()
        plt.rcParams["figure.figsize"] = old_sz
        f.write("<img src=\"" + storeTime + ".png\">")
        f.write("</body>")
        f.write("</html>")
    factory_size = len(last_map.entities)+1
    module_score = np.sum([(m.repeats-1) * len(m.configuration) for m in modules]) / np.sum([len(m.machines) for m in modules])
    n_markers = last_map.count_markers()
    return factory_size, last_map.level, obsolete.shape[0]/factory_size, n_markers/factory_size, module_score, n_turns

async def handle_user(user_id):
    stats = [("factory_size", 0), ("level", 1), ("obsolete", 2), ("label", 3), ("modules", 4), ("belt_spaghetti", 5)]
    path = "../../results/players"
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + "/" + user_id
    if not os.path.exists(path):
        os.mkdir(path)
    if os.path.exists(path + "/data.csv"):
        with open(path + "/data.csv", "r") as f:
            stream = csv.reader(f, delimiter=",")
            header = next(stream)
            if np.all([head == expected for head, (expected, _) in zip(header[1:], stats)]):
                print("loading saved data for " + user_id)
                points = []
                for row in stream:
                    points.append((row[0], tuple([float(x) for x in row[1:]])))
            count = len(points)
            first_time = points[0][0]
            last_time = points[-1][0]
            [row] = await database.fetch_all(query="SELECT * FROM Saves WHERE userId = :userId AND storeTime = :storeTime", values={"userId": user_id, "storeTime": last_time})
            last_save = analysis.decompress_save_data(row[2])
            last_save_compressed = row[2]
    else:
        rows = await database.fetch_all(query="SELECT * FROM Saves WHERE userId = :userId", values={"userId":user_id})
        count = 0
        first_time = None
        last_time = None
        last_save = None
        last_save_compressed = None
        points = []
        rows = list(rows)
        rows.sort(key = lambda x: x[1], reverse=False)
        #rows = rows[0:10] + [rows[-1]] # TODO remove this when final script is ready
        saves = [(user_id, storeTime, analysis.decompress_save_data(compressed_save_data), compressed_save_data) for user_id, storeTime, compressed_save_data in rows]
        ok_saves = [i == 0 or len(x[2]["dump"]["entities"]) > 1 for i, x in enumerate(saves)]
        rows = list(np.array(rows)[ok_saves])
        saves = list(np.array(saves)[ok_saves])
        times = [storeTime for _, storeTime, _ in rows]
        befores = times[1:] + [None]
        afters = [None] + times[:-1]
        save_list = zip(saves, befores, afters)
        with open(path + "/data.csv", "w") as f:
            stream = csv.writer(f, delimiter=",")
            stream.writerow(["time"] + [head for head, _ in stats])
            for i, ((_, storeTime, save_data, save_data_compressed), before, after) in enumerate(save_list):
                print(storeTime, user_id)
                points.append((storeTime, await handle_save(user_id, path, storeTime, save_data, before, after)))
                stream.writerow([points[-1][0]] + list(points[-1][1]))
                count += 1
                if first_time is None or first_time > storeTime:
                    first_time = storeTime
                if last_time is None or last_time < storeTime:
                    last_time = storeTime
                    last_save = save_data
                    last_save_compressed = save_data_compressed
        print(count, user_id)
    event_rows = await database.fetch_all(query="SELECT * FROM Events WHERE userId = :userId", values={"userId":user_id})
    event_rows = list(event_rows)
    event_rows.sort(key=lambda x: x[1], reverse=False)
    closed_notifications = False
    for _, storeTime, eventData in event_rows:
        if "obnoxious notifications" in eventData and "closed" in eventData:
            closed_notifications = True
            break
    with open(path + "/index.html", "w") as f:
        f.write("<html><head><title>Player results</title></head>")
        f.write("<body>")
        f.write("<h1>Results for " + user_id + "</h1>")
        f.write("<p>Final map:</p>")
        print("Initializing map representation")
        last_map = analysis.MapState(last_save)
        print("Marking obsolete")
        obsolete = np.array(last_map.mark_obsolete())
        print("Identifying modules")
        modules, machines_in_modules = last_map.identify_modules()
        print("Identifying paths")
        paths, n_turns = last_map.identify_connecting_paths()
        print("Rendering")
        analysis.render_plt(last_map)
        plt.savefig(path + "/final_map.png")
        plt.close()
        old_sz = plt.rcParams["figure.figsize"]
        plt.rcParams["figure.figsize"] = (last_map.w * 0.33 + 1,last_map.h * + 0.33 + 1)
        analysis.render_plt(last_map)
        plt.savefig(path + "/final_map_zoom.png")
        plt.close()
        plt.rcParams["figure.figsize"] = old_sz
        f.write("<img src=\"final_map.png\">")
        f.write("<p><a href=\"final_map_zoom.png\">Full size</a></p>")
        f.write("<h2>Stats over playtime</h2>")
        for stat_name, stat_ix in stats:
            plt.plot([dp.parse(t).timestamp() for t, _ in points], [stat[stat_ix] for _, stat in points])
            plt.xlabel("time")
            plt.ylabel(stat_name)
            plt.savefig(path + "/stat_" +  stat_name + ".png")
            plt.close()
            f.write("<img src=\"stat_" + stat_name + ".png\">")
        f.write("</body")
        f.write("</html>")
    with open(path + "/savegame.db", "w") as f:
        f.write(last_save_compressed)
    return count, first_time, points, last_map.reorg_score, last_map.count_markers(), closed_notifications

def print_corr_mat(labels, variables):
    variables = np.array(variables)
    corrs = np.corrcoef(variables)
    print("\\begin{table}[ht]")
    print("\\begin{tabular}{l|" + "r" * len(labels) + "}")
    print("& " + " & ".join(labels) + "\\\\ \\hline")
    for i in range(len(labels)):
        print(labels[i] + " & " + " & ".join([str(np.round(c, 2)) for c in corrs[i]]) + " \\\\")
    print("\\end{tabular}")
    print("\\end{table}")

def print_cross_corr_mat(labels1, variables1, labels2, variables2):
    variables1 = np.array(variables1)
    variables2 = np.array(variables2)
    corrs = np.corrcoef(np.concatenate([variables1, variables2]))[:len(labels1), len(labels1):]
    print("\\begin{table}[ht]")
    print("\\begin{tabular}{l|" + "r" * len(labels2) + "}")
    print("& " + " & ".join(labels2) + "\\\\ \\hline")
    for i in range(len(labels1)):
        print(labels1[i] + " & " + " & ".join([str(np.round(c, 2)) for c in corrs[i]]) + " \\\\")
    print("\\end{tabular}")
    print("\\end{table}")
def print_mean_mat(label_vars, label_groups, groups_mu, groups_sigma):
    groups_mu = np.array(groups_mu)
    groups_sigma = np.array(groups_sigma)
    print("\\begin{table}[ht]")
    print("\\begin{tabular}{l|" + "c" * len(label_vars) + "}")
    print("& " + " & ".join(label_vars) + "\\\\")
    for i in range(len(label_groups)):
        print("\\hline")
        print(label_groups[i] + " & " + " & ".join([str(np.round(c, 2)) for c in groups_mu[i]]) + " \\\\")
        print(" & " + " & ".join(["(" + str(np.round(c, 2)) + ")" for c in groups_sigma[i]]) + " \\\\")
    print("\\end{tabular}")
    print("\\end{table}")
def factor(variables, align=True):
    variables = np.array(variables).T
    variables = variables / np.std(variables, axis=0)
    if align:
        variables = variables * np.sign(np.corrcoef(variables.T)[0])
    return np.mean(variables, axis=1)

class Table:
    def __init__(self):
        self.columns = []
        self.data = []
    def make_col(self, name):
        identifier = len(self.columns)
        self.columns.append(name)
        return identifier
    def get_col(self, name):
        return self.columns.index(name)
    def add(self, **kwargs):
        point = []
        for name in self.columns:
            point.append(kwargs[name])
        self.data.append(point)
    def to_array(self):
        return np.array(self.data).reshape((len(self.data), len(self.columns)))

def skip_until(col, name):
    skip = True
    def func(it):
        nonlocal skip
        if name == it[col]:
            skip = False
        return skip
    return func
def load_survey(file, skip):
    with open(file, "r") as f:
        stream = csv.reader(f, delimiter=",")
        header = next(stream)
        resps = {}
        for row in stream:
            resp = {}
            for i, h in enumerate(header):
                resp[h] = row[i]
            if skip(resp):
                continue
            resps[row[0]] = resp
    return header, resps

async def main():
    # the first few responses were just used to test the system, so skipping until the appropriate session
    header, test_users = load_survey("../../shapez_io_personality_test.csv", skip_until("session", "8TETHMKXgpKk5cwasWRHE0Js7X8XF9rhW9OPX4e9hem2QHyYWhpNE571D5c6z_52"))
    post_header, post_test_users = load_survey("../../followup_shapez.csv", skip_until("session", "Xc88W8kZaWiIMcevg5LqiLPVnjp9l0AMkLs09hmvhHD_bZ13_OfJv9dLc_wXqc0k"))
    post_test_ids = [uid for uid in post_test_users.keys() if post_test_users[uid]["ended"] != ""]
    post_vars = np.array([[int(post_test_users[uid][header]) for header in post_header if "pref" in header] for uid in post_test_ids])
    post_vars = np.array([np.mean(post_vars[:, [0, 1, 2]], axis=1), np.mean(post_vars[:, [3, 4, 5]], axis=1), np.mean(post_vars[:, [6, 7, 8]], axis=1)]).T
    post_vars = np.concatenate([post_vars, post_vars[:, [1]]-post_vars[:, [0]]], axis=1)
    traits = ["organization", "perfectionism", "diligence", "prudence", "honesty", "emotionality", "extraversion", "agreeableness", "openness"]
    ok_users = [uid for uid in test_users.keys() if test_users[uid]["ended"] != ""]
    personality_matrix = []
    for trait in traits:
        trait_items = np.array([[int(test_users[uid][item]) for item in header if trait in item] for uid in ok_users])
        signs = np.array([-1 if "R" in item else 1 for item in header if trait in item])
        trait_items = signs * trait_items + -(signs - 1)//2 * 6
        sum_score = np.sum(trait_items, axis=1)
        var_items = np.sum(np.var(trait_items, axis=0))
        var_score = np.var(sum_score)
        alpha = trait_items.shape[1] / (trait_items.shape[1]-1) * (1 - var_items / var_score)
        print("Alpha for " + trait + ": " + str(np.round(alpha, 2)))
        print(np.round(np.corrcoef(trait_items.T), 2))
        personality_matrix.append(sum_score / trait_items.shape[1])
    personality_matrix = np.array(personality_matrix).T
    subset = np.array([uid in post_test_ids for uid in ok_users])
    reorder = np.array([post_test_ids.index(uid) for uid in np.array(ok_users)[subset]])
    print_cross_corr_mat(traits, personality_matrix[subset].T, ["disorderly", "orderly", "control", "difference"], post_vars[reorder].T)
    experience_vars = np.array([[int(test_users[uid][header]) for header in header if "experience" in header] for uid in ok_users])
    await database.connect()
    rows = await database.fetch_all(query="SELECT * FROM Saves")
    userIds = {}
    for i, (userId, storeTime, compressed_save_data) in enumerate(rows):
        if not userId in userIds:
            userIds[userId] = 0
        userIds[userId] += 1
    userIds = set([userId for userId, count in userIds.items() if count >= 8])
    print(userIds)
    print(len(userIds))
    text_data = Table()
    data = Table()
    personality_data = Table()
    ct_id = text_data.make_col("player")
    ct_time = text_data.make_col("time")
    cn_time = data.make_col("time")
    cn_save_count = data.make_col("saves")
    cn_level = data.make_col("level")
    cn_size = data.make_col("machines")
    cn_reorg = data.make_col("reorg")
    cn_obsolete = data.make_col("obsolete")
    cn_labels = data.make_col("labels")
    cn_modules = data.make_col("tiled")
    cn_spaghetti = data.make_col("straight")
    cn_notifications = data.make_col("notifs")
    for trait in traits:
        personality_data.make_col(trait)
    for user_id in userIds:
        if not user_id in test_users:
            continue
        try:
            count, storeTime, points, reorg_score, label, closed_notifications = await handle_user(user_id)
        except:
            continue
        last_point = points[-1][1]
        game_level = last_point[1]
        factory_size = last_point[0]
        obsolete = last_point[2]
        #label = last_point[3]
        modules = last_point[4]
        spaghetti = last_point[5]
        text_data.add(player=user_id, time=storeTime)
        data.add(time=dp.parse(storeTime).timestamp(), saves=count, level=game_level, machines=factory_size, reorg=reorg_score, obsolete=obsolete, labels=label, tiled=modules, straight=-spaghetti, notifs=closed_notifications)
        kwargs = {}
        respondent = test_users[user_id]
        for trait in traits:
            score = 0
            n_items = 0
            for key in respondent.keys():
                if trait in key:
                    item = int(respondent[key])
                    if "R" in key:
                        item = 6 - item
                    score += item
                    n_items += 1
            kwargs[trait] = score / n_items
        personality_data.add(**kwargs)
        #if len(data.data) > 5:
        #    break
    text_data, text_data_table = text_data.to_array(), text_data
    data, data_table = data.to_array(), data
    personality_data, personality_data_table = personality_data.to_array(), personality_data
    #print(data[:, COL_LEVEL])
    oks = np.logical_and(data[:, cn_level] > 7, data[:, cn_size] > 60)
    data = data[oks]
    text_data = text_data[oks]
    personality_data = personality_data[oks]
    subset = np.array([uid in text_data[:, ct_id] for uid in ok_users])
    exp_overall = np.array([experience_vars[:, 0] == 1, experience_vars[:, 0] == 2, experience_vars[:, 0] == 3, experience_vars[:, 0] == 4]).T
    print_mean_mat(traits, ["all participants", "selected participants"], [np.mean(personality_matrix, axis=0), np.mean(personality_matrix[subset], axis=0)], [np.std(personality_matrix, axis=0), np.std(personality_matrix[subset], axis=0)])
    print_mean_mat(["shapez.io", "Factorio", "Dyson Sphere Program"], ["all", "selected"], [np.mean(experience_vars[:, 1:], axis=0), np.mean(experience_vars[subset, 1:], axis=0)], [np.zeros(3), np.zeros(3)])
    print_mean_mat(["None", "Almost none", "Some", "A lot"], ["all", "selected"], [np.mean(exp_overall, axis=0), np.mean(exp_overall[subset, :], axis=0)], [np.zeros(4), np.zeros(4)])
    do_log = ["saves", "machines"]
    fig, ax = plt.subplots(3, 3, figsize=(14, 18))
    for i in range(9):
        x = i % 3
        y = i // 3
        col = i + 1
        ax[y, x].set_title(data_table.columns[col])
        if data_table.columns[col] in do_log:
            ax_loghist(ax[y, x], data[:, col])
        else:
            ax[y, x].hist(data[:, col])
        ax[y, x].set_xlabel({"saves":"saves", "level":"level", "machines":"count", "labels":"count"}.get(i) or "measure")
        ax[y, x].set_ylabel("count")
    fig.subplots_adjust(hspace=0.35)
    plt.show()
    print(data.shape)
    #print(data[:, COL_LEVEL])
    #exit()
    print_corr_mat(["log(saves)", "log(machines)", "level"], [np.log(data[:, cn_save_count]), np.log(data[:, cn_size]), data[:, cn_level]])
    progress = factor([np.log(data[:, cn_save_count]), np.log(data[:, cn_size]), data[:, cn_level]])
    order = factor([data[:, cn_reorg], -np.log(1+data[:, cn_obsolete]), data[:, cn_labels], data[:, cn_modules], data[:, cn_spaghetti], data[:, cn_notifications]], align=False)
    print_corr_mat(["progress", "overall beh", "reorg_score", "log(obsolete)", "label", "modules", "straight", "notifications"]
                , [progress, order, data[:, cn_reorg], np.log(1+data[:, cn_obsolete]), data[:, cn_labels], data[:, cn_modules], data[:, cn_spaghetti], data[:, cn_notifications]])
    print(personality_data.shape, len(list(personality_data.T)))
    print(len(traits))
    print_corr_mat(["progress", "overall beh"] + traits, [progress, order] + list(personality_data.T))
    print_cross_corr_mat(["progress", "overall beh", "reorg_score", "log(obsolete)", "label", "modules", "straight", "notifications"]
                        ,[progress, order, data[:, cn_reorg], np.log(1+data[:, cn_obsolete]), data[:, cn_labels], data[:, cn_modules], data[:, cn_spaghetti], data[:, cn_notifications]]
                        ,traits, list(personality_data.T))
    print("\\begin{tabular}{l|" + "r" * (len(personality_data_table.columns) - 5) + "rr}")
    print("& " + " & ".join(data_table.columns[4:-1]) + " & \\(R^2\\) & \\(R^2\\) (LOOCV) \\\\ \\hline")
    for i, trait in enumerate(personality_data_table.columns):
        model = LinearRegression()
        preds = []
        for loo in range(data.shape[0]+1):
            included = np.arange(0, data.shape[0]) != loo
            model.fit(data[included][:, 4:-1], personality_data[included][:, i])
            if loo != data.shape[0]:
                preds.append(model.predict(data[loo, 4:-1].reshape((1, -1))).reshape(()))
        preds = np.array(preds)
        sse = np.sum((preds - personality_data[:, i])**2)
        var = personality_data.shape[0] * np.var(personality_data[:, i])
        adj_r2 = 1-sse/var
        model.fit(data[:, 4:-1], personality_data[:, i])
        sse = np.sum((model.predict(data[:, 4:-1]) - personality_data[:, i])**2)
        var = personality_data.shape[0] * np.var(personality_data[:, i])
        naive_r2 = 1-sse/var
        standardized_coefficients = model.coef_ * np.std(data[:, 4:-1], axis=0) / np.std(personality_data[:, i])
        print(trait + " & " + " & ".join([str(np.round(x, 2)) for x in standardized_coefficients]) + " & " + str(np.round(naive_r2, 2)) + " & " + str(np.round(adj_r2, 2)) + " \\\\")
    print("\\end{tabular}")
    exit()
    path = "../../results"
    if not os.path.exists(path):
        os.mkdir(path)
    with open("../../results/index.html", "w") as f:
        f.write("<html><head><title>Overall results</title></head>")
        f.write("<body>")
        f.write("<h1>Overall results</h1>")
        plot_loghist(data[:, cn_save_count])
        plt.title("Amount of savegame data points per player")
        plt.xlabel("Number of saves")
        plt.ylabel("Count of players")
        plt.savefig("../../results/save_hist.png")
        plt.close()
        f.write("<img src=\"save_hist.png\"><br>")
        plt.hist(order)
        plt.xlabel("Behavioral orderliness")
        plt.ylabel("Number of players")
        plt.savefig("../../results/order_hist.png")
        plt.close()
        f.write("<img src=\"order_hist.png\"><br>")
        f.write("<p>Players:</p>")
        f.write("<ul>")
        for i in range(data.shape[0]):
            user_id = text_data[i, ct_id]
            f.write("<li><a href=\"players/" + user_id + "/index.html\">" + user_id[0:6] + "@" + text_data[i, ct_time][5:19] + "</a>, " + str(data[i, cn_save_count]) + " saves, " + str(data[i, cn_size]) + " machines, level " + str(data[i, cn_level]))
        f.write("</ul>")
        f.write("<h1>Variables</h1>")
        for i, title in enumerate(data_table.columns):
            f.write("<h2>" + title + "</h2>")
            f.write("<ul>")
            f.write("<li>Highest player: " + text_data[np.argmax(data[:, i]), ct_id])
            f.write("<li>Lowest player: " + text_data[np.argmin(data[:, i]), ct_id])
            f.write("</ul>")
            plt.title(title)
            plt.hist(data[:, i])
            plt.savefig("../../results/hist_" + title +".png")
            plt.ylabel("Count of players")
            plt.xlabel(title)
            plt.close()
            f.write("<img src=\"hist_" + title + ".png\"><br>")
        f.write("</body>")
        f.write("</html>")


asyncio.get_event_loop().run_until_complete(main())