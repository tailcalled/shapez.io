import { STOP_PROPAGATION } from "../../../core/signal";
import { Vector } from "../../../core/vector";
import { enumMouseButton } from "../../camera";
import { BaseHUDPart } from "../base_hud_part";
import { MarkerComponent } from "../../components/marker";
import { MarkerSystem } from "../../systems/marker";
import { MetaMarkerBuilding } from "../../buildings/marker";
import { DrawParameters } from "../../../core/draw_parameters";
import { Rectangle } from "../../../core/rectangle";

export class HUDMarkerEdit extends BaseHUDPart {
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
            const contents = this.root.map.getLayerContentXY(tile.x, tile.y, "regular");
            if (contents) {
                const markerComp = contents.components.Marker;
                if (markerComp) {
                    markerComp.openSettings(false, contents);
                    return STOP_PROPAGATION;
                }
            }
            const world = this.root.camera.screenToWorld(pos);
            for (let pair of this.root.systemMgr.systems.marker.markedAreas) {
                if (pair.rect.containsPoint(world.x, world.y)) {
                    let dragLeft = world.x < pair.rect.x + 16;
                    let dragRight = world.x > pair.rect.x + pair.rect.w - 16;
                    let dragBot = world.y < pair.rect.y + 16;
                    let dragTop = world.y > pair.rect.y + pair.rect.h - 16;
                    if (dragLeft || dragRight || dragBot || dragTop) {
                        this.dragging = { l: dragLeft, r: dragRight, b: dragBot, t: dragTop, e: pair.ent };
                        return STOP_PROPAGATION;
                    }
                }
            }
        }
    }
    upPostHandler(pos, button) {
        if (this.dragging) {
            let markerComp = this.dragging.e.components.Marker;
            let staticComp = this.dragging.e.components.StaticMapEntity;
            let epos = staticComp.getTileSpaceBounds().getCenter().toWorldSpace();
            let rect = markerComp.rect;
            let x0 = Math.round((rect.x - epos.x - 16) / 32);
            let y0 = Math.round((rect.y - epos.y - 16) / 32);
            let x1 = Math.round((rect.x + rect.w - epos.x + 16) / 32);
            let y1 = Math.round((rect.y + rect.h - epos.y + 16) / 32);
            markerComp.x0 = x0 + 1;
            markerComp.y0 = y0 + 1;
            markerComp.x1 = x1 - 1;
            markerComp.y1 = y1 - 1;
            markerComp.rect = null;
        }
        this.dragging = null;
        return;
    }
    movePreHandler(vec) {
        if (this.dragging) {
            let markerComp = this.dragging.e.components.Marker;
            let staticComp = this.dragging.e.components.StaticMapEntity;
            let epos = staticComp.getTileSpaceBounds().getCenter().toWorldSpace();
            let rect = markerComp.rect;
            let pos = this.root.camera.screenToWorld(vec);
            let newX0 = rect.x;
            if (this.dragging.l) {
                newX0 = Math.min(epos.x - 8, pos.x);
            }
            let newX1 = rect.x + rect.w;
            if (this.dragging.r) {
                newX1 = Math.max(epos.x + 8, pos.x);
            }
            let newY0 = rect.y;
            if (this.dragging.b) {
                newY0 = Math.min(epos.y - 8, pos.y);
            }
            let newY1 = rect.y + rect.h;
            if (this.dragging.t) {
                newY1 = Math.max(epos.y + 8, pos.y);
            }
            markerComp.rect = new Rectangle(newX0, newY0, newX1 - newX0, newY1 - newY0);
        }
    }
    /**
     * Should draw the hud
     * @param {DrawParameters} parameters
     */
    draw(parameters) {}
}
