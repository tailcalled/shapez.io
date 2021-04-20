/* typehints:start */
import { Application } from "../application";
/* typehints:end */

import { createLogger } from "../core/logging";
import { BasicSerializableObject, types } from "../savegame/serialization";
import { Signal } from "../core/signal";
import { Vector } from "../core/vector";

const logger = createLogger("data_connection");

export class DataConnection {
    constructor(app) {
        /** @type {Application} */
        this.app = app;

        /** @type {WebSocket|null} */
        this.socket = null;

        /** @type {Array<Array<string>>} */
        this.signals = [];

        Signal.data_stream = this;
    }

    initialize() {
        let userId = this.app.settings.getUserId();
        return new Promise((resolve, reject) => {
            let host = window.location.hostname;
            if (host == "localhost") {
                host = "ws://" + host + ":3006";
            } else {
                host = "wss://" + host;
            }
            let socket = new WebSocket(host + "/orderliness/game_data");
            socket.onopen = function (event) {
                socket.send(userId);
                resolve(socket);
            };
            socket.onerror = function (event) {
                alert("Could not connect to server");
                reject(event);
            };
        }).then(socket => (this.socket = socket));
    }

    collectDataAsync(data) {
        return new Promise((resolve, reject) => {
            this.socket.send(JSON.stringify(this.signals));
            this.signals = [];
            this.socket.send(data);
            resolve(null);
        });
    }
    signalDispatch(name, args) {
        let parts = [name, new Date().getTime()];
        for (let arg of args) {
            if (arg instanceof BasicSerializableObject) {
                parts.push(arg.serialize());
            } else if (typeof arg == "string" || typeof arg == "number") {
                parts.push(arg);
            } else if (arg instanceof Vector) {
                parts.push(arg.x, arg.y);
            } else if (Array.isArray(arg)) {
                parts.push(...arg);
            } else {
                alert(arg);
            }
        }
        this.signals.push(parts);
    }
}
