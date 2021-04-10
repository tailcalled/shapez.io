import { types } from "../../savegame/serialization";
import { BaseItem } from "../base_item";
import { Component } from "../component";
import { typeItemSingleton } from "../item_resolver";
import { ColorItem, COLOR_ITEM_SINGLETONS } from "../items/color_item";
import { ShapeItem } from "../items/shape_item";
import { Entity } from "../entity";
import { FormElementInput, FormElementItemChooser, FormElementCheckbox } from "../../core/modal_dialog_forms";
import { DialogWithForm } from "../../core/modal_dialog_elements";
import { T } from "../../translations";
import { type } from "os";

export class MarkerComponent extends Component {
    static getId() {
        return "Marker";
    }

    static getSchema() {
        return {
            title: types.string,
            x0: types.int,
            y0: types.int,
            x1: types.int,
            y1: types.int,
        };
    }

    /**
     * @param {object} param0
     */
    constructor() {
        super();
        this.title = "";
        this.x0 = -1;
        this.y0 = -1;
        this.x1 = 1;
        this.y1 = 1;
        this.rect = null;
    }

    setTitle(title) {
        this.title = title;
    }

    /**
     *
     * @param {boolean} isPlacing true if the marker is getting placed, false if the marker is just getting edited
     * @param {Entity} entity entity that is getting configured
     */
    openSettings(isPlacing, entity) {
        const self = this;
        const markerNameInput = new FormElementInput({
            id: "markerName",
            label: null,
            placeholder: "",
            defaultValue: isPlacing ? "" : entity.components.Marker.title,
            validator: val => val.length > 0 && val.length < 71,
        });
        const dialog = new DialogWithForm({
            app: entity.root.app,
            title: true ? T.dialogs.placeMarker.titleEdit : T.dialogs.placeMarker.title,
            desc: T.dialogs.placeMarker.desc,
            formElements: [markerNameInput],
            buttons: ["cancel", "ok:good"],
        });
        entity.root.hud.parts.dialogs.internalShowDialog(dialog);
        dialog.buttonSignals.ok.add(() => {
            self.setTitle(markerNameInput.getValue());
        });
        if (isPlacing) {
            dialog.buttonSignals.cancel.add(() => entity.root.logic.tryDeleteBuilding(entity));
        }
    }
}
