import { STOP_PROPAGATION } from "../../../core/signal";
import { Vector } from "../../../core/vector";
import { enumMouseButton } from "../../camera";
import { BaseHUDPart } from "../base_hud_part";
import { MarkerComponent } from "../../components/marker";
import { MarkerSystem } from "../../systems/marker";
import { MetaMarkerBuilding } from "../../buildings/marker";
import { DrawParameters } from "../../../core/draw_parameters";
import { Rectangle } from "../../../core/rectangle";

export class HUDResourceNodeEdit extends BaseHUDPart {
    initialize() {
        this.root.camera.downPreHandler.add(this.downPreHandler, this);
        this.root.camera.upPostHandler.add(this.upPostHandler, this);
        this.root.camera.movePreHandler.add(this.movePreHandler, this);
        this.dragging = null;
    }

    /**
     * @param {Vector} pos
     * @param {enumMouseButton} button
     */
    downPreHandler(pos, button) {
        if (button === enumMouseButton.left) {
            const tile = this.root.camera.screenToWorld(pos).toTileSpace();
            if (!this.root.map.getLayerContentXY(tile.x, tile.y, "regular")) {
                const contents = this.root.map.getLowerLayerContentXY(tile.x, tile.y);
                if (contents) {
                    this.dragging = { x: tile.x, y: tile.y };
                    this.decideValidPlacements(tile);
                    return STOP_PROPAGATION;
                }
            }
        }
    }
    decideValidPlacements(tile) {
        const type = this.root.map.getLowerLayerContentXY(tile.x, tile.y);
        // OK SO, here's the thing
        // we want the player to be able to move tiles around to make the mining posts more orderly
        // BUT we don't want them to just make the whole playing task trivial by moving ores to where they need them
        // therefore we need to limit their range to only moving tiles around near places where there were already tiles.
        // we could keep track of where the tiles were to begin with to make designated areas where the tiles can exist, but
        // that would require a lot more modification. instead, let's just say that any one tile can only be moved to a place
        // close to a close tile; that should be plenty.

        // compute the minimum distance for every nearby tile to a nearby ore using bfs
        let queue = [];
        /** @type {Map<number, Map<number, number>>} */
        let values = new Map();
        for (let x = tile.x - 4; x <= tile.x + 5; x++) {
            for (let y = tile.y - 4; y <= tile.y + 5; y++) {
                const local = this.root.map.getLowerLayerContentXY(x, y);
                if ((x != tile.x || y != tile.y) && type == local) {
                    queue.push({ x: x, y: y, d: 0 });
                }
            }
        }
        while (queue.length > 0) {
            let it = queue.shift(); // this is O(n) time, but using pop would be a dfs instead, which ends up risking exponential time.
            if (!values.has(it.x)) values.set(it.x, new Map());
            if (values.get(it.x).has(it.y)) continue;
            values.get(it.x).set(it.y, it.d);
            if (it.d < 3) {
                for (let xo = -1; xo <= 1; xo++) {
                    for (let yo = -1; yo <= 1; yo++) {
                        if (xo * xo + yo * yo == 1) {
                            queue.push({ x: it.x + xo, y: it.y + yo, d: it.d + 1 });
                        }
                    }
                }
            }
        }

        // we don't actually need the distances, but the found positions can then be used as valid areas
        this.valid = values;
    }
    upPostHandler(pos, button) {
        if (this.dragging) {
            if (this.isValidDropPos(pos)) {
                const target = this.root.camera.screenToWorld(pos).toTileSpace();
                const source = this.root.map.getLowerLayerContentXY(this.dragging.x, this.dragging.y);
                this.root.map.setLowerLayerContentXY(target.x, target.y, source);
                this.root.map.setLowerLayerContentXY(this.dragging.x, this.dragging.y, null);
            }
            this.dragging = null;
            return STOP_PROPAGATION;
        }
    }
    movePreHandler(vec) {
        this.cursor = vec;
    }

    isValidDropPos(pos) {
        const target = this.root.camera.screenToWorld(pos).toTileSpace();
        return (
            !this.root.map.getLowerLayerContentXY(target.x, target.y) &&
            !this.root.map.getLayerContentXY(target.x, target.y, this.root.currentLayer) &&
            this.valid.has(target.x) &&
            this.valid.get(target.x).has(target.y)
        );
    }
    /**
     * Should draw the hud
     * @param {DrawParameters} parameters
     */
    draw(parameters) {
        if (this.dragging) {
            const source = this.root.map.getLowerLayerContentXY(this.dragging.x, this.dragging.y);
            const pos = this.root.camera.screenToWorld(this.cursor);
            for (let x of this.valid.keys()) {
                for (let y of this.valid.get(x).keys()) {
                    parameters.context.globalAlpha = 0.1;
                    parameters.context.fillStyle = "#ffff00";
                    parameters.context.fillRect(x * 32, y * 32, 32, 32);
                    parameters.context.globalAlpha = 1;
                }
            }
            if (this.isValidDropPos(this.cursor)) {
                const tile = pos.snapWorldToTile();
                source.drawItemCenteredClipped(tile.x + 16, tile.y + 16, parameters);
            }
            source.drawItemCenteredClipped(pos.x, pos.y, parameters);
        }
    }
}
