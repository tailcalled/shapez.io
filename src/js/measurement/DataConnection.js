/* typehints:start */
import { Application } from "../application";
/* typehints:end */

import { createLogger } from "../core/logging";

const logger = createLogger("data_connection");

export class DataConnection {
    constructor(app) {
        this.app = app;

        /** @type {WebSocket|null} */
        this.socket = null;
    }

    initialize() {
        let userId = this.app.settings.getUserId();
        return new Promise((resolve, reject) => {
            let host = window.location.hostname;
            if (host == "localhost") {
                host = host + ":3006";
            }
            let socket = new WebSocket("ws://" + host + "orderliness/game_data");
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
            this.socket.send(data);
            resolve(null);
        });
    }
}
