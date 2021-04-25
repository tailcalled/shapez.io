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

        /** @type {number} */
        this.packet_id = new Date().getTime();

        /** @type {Array<[number,string]>} */
        this.packets = [];

        this.timeout_id = null;
    }

    initialize() {
        return this.connect();
    }
    reconnect(event) {
        this.connect();
    }
    connect() {
        let userId = this.app.settings.getUserId();
        let self = this;
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
                alert(
                    "Could not connect to server; please contact survey administration as this can result in data loss."
                );
                reject(event);
            };
            socket.onclose = function (event) {
                self.reconnect(event);
            };
            socket.onmessage = function (event) {
                self.confirm(event);
            };
        }).then(socket => (this.socket = socket));
    }

    trySend(msg) {
        this.packets.push([this.packet_id++, msg]);
        if (this.timeout_id) {
            clearTimeout(this.timeout_id);
        }
        this.timeout_id = setTimeout(() => {
            this.timeout_id = null;
            this.flushPackets();
        }, 500);
    }
    flushPackets() {
        for (let packet of this.packets) {
            this.socket.send(packet[0].toString());
            this.socket.send(packet[1]);
        }
    }
    /** @param {MessageEvent} event */
    confirm(event) {
        let confirmation_id = parseFloat(event.data);
        this.packets = this.packets.filter(packet => packet[0] <= confirmation_id);
    }

    collectDataAsync(data) {
        return new Promise((resolve, reject) => {
            this.trySend("SAVE:" + data);
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
        this.trySend("EVENT:" + JSON.stringify(parts));
    }
}
