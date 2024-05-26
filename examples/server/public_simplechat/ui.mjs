//@ts-check
// Helpers to work with html elements
// by Humans for All
//


/**
 * Set the class of the children, based on whether it is the idSelected or not.
 * @param {HTMLDivElement} elBase
 * @param {string} idSelected
 * @param {string} classSelected
 * @param {string} classUnSelected
 */
export function el_children_config_class(elBase, idSelected, classSelected, classUnSelected="") {
    for(let child of elBase.children) {
        if (child.id == idSelected) {
            child.className = classSelected;
        } else {
            child.className = classUnSelected;
        }
    }
}

/**
 * Create button and set it up.
 * @param {string} id
 * @param {(this: HTMLButtonElement, ev: MouseEvent) => any} callback
 * @param {string | undefined} name
 * @param {string | undefined} innerText
 */
export function el_create_button(id, callback, name=undefined, innerText=undefined) {
    if (!name) {
        name = id;
    }
    if (!innerText) {
        innerText = id;
    }
    let btn = document.createElement("button");
    btn.id = id;
    btn.name = name;
    btn.innerText = innerText;
    btn.addEventListener("click", callback);
    return btn;
}

/**
 * Create a para and set it up. Optionaly append it to a passed parent.
 * @param {string} text
 * @param {HTMLElement | undefined} elParent
 * @param {string | undefined} id
 */
export function el_create_append_p(text, elParent=undefined, id=undefined) {
    let para = document.createElement("p");
    para.innerText = text;
    if (id) {
        para.id = id;
    }
    if (elParent) {
        elParent.appendChild(para);
    }
    return para;
}

/**
 * Create a para and set it up. Optionaly append it to a passed parent.
 * @param {string} id
 * @param {{true: string, false: string}} texts
 * @param {boolean} defaultValue
 * @param {function(boolean):void} cb
 */
export function el_create_boolbutton(id, texts, defaultValue, cb) {
    let el = document.createElement("button");
    el["xbool"] = defaultValue;
    el["xtexts"] = structuredClone(texts);
    el.innerText = el["xtexts"][String(defaultValue)];
    if (id) {
        el.id = id;
    }
    el.addEventListener('click', (ev)=>{
        el["xbool"] = !el["xbool"];
        el.innerText = el["xtexts"][String(el["xbool"])];
        cb(el["xbool"]);
    })
    return el;
}

/**
 * @param {string} id
 * @param {string[]} options
 * @param {string} defaultOption
 * @param {function(string):void} cb
 */
export function el_create_select(id, options, defaultOption, cb) {
    let el = document.createElement("select");
    el["xselected"] = defaultOption;
    el["xoptions"] = structuredClone(options);
    for(let cur of options) {
        let op = document.createElement("option");
        op.value = cur;
        op.innerText = cur;
        if (cur == defaultOption) {
            op.selected = true;
        }
        el.appendChild(op);
    }
    if (id) {
        el.id = id;
        el.name = id;
    }
    el.addEventListener('click', (ev)=>{
        let target = /** @type{HTMLSelectElement} */(ev.target);
        console.log("DBUG:UI:Select:", id, ":", target.value);
        cb(target.value);
    })
    return el;
}
